"""Provenance, official validation and full distributed training utilities."""
import json
from pathlib import Path
import random
import numpy as np
import torch
import torch.distributed as dist
from focusseg_r1.full_data import file_digest


def identity():
    root = Path(__file__).resolve().parents[1]
    paths = list((root / "seg_lora").glob("*.py")) + [root / p for p in (
        "local_scripts/seg_lora_run.py", "modeling_sa2va_pro.py", "configuration_sa2va_pro.py",
        "sam2.py", "sa2va_checkpoint_utils.py", "sa2va_pro_utils.py", "focusseg_r1/full_data.py",
        "focusseg_r1/refvos_data.py", "focusseg_r1/full_runtime.py", "local_scripts/evaluate_refsav_jf.py")]
    return {str(p.relative_to(root)): file_digest(p) for p in paths}


def rng_state():
    n = np.random.get_state()
    return {"python": random.getstate(), "numpy": [n[0], n[1].tolist(), n[2], n[3], n[4]],
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()}


def restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"].cpu())
    torch.cuda.set_rng_state(state["cuda"].cpu())


def gather(value):
    result = [None] * dist.get_world_size()
    dist.all_gather_object(result, value)
    return result


def validation_summary(rows):
    if not rows or len({(r["video_id"], r["exp_id"]) for r in rows}) != len(rows):
        raise ValueError("Missing/duplicate validation expressions")
    groups = {}
    for key, subset in (("Overall", rows), ("Long", [r for r in rows if r["type_id"] == 0]),
                        ("Short", [r for r in rows if r["type_id"] == 1])):
        if not subset:
            raise ValueError(f"Empty validation group: {key}")
        groups[key] = {k: sum(r[k] for r in subset) / len(subset) for k in ("J", "F", "J&F")}
        groups[key]["N"] = len(subset)
    groups["PaperBalanced"] = {k: (groups["Long"][k] + groups["Short"][k]) / 2 for k in ("J", "F", "J&F")}
    return groups


@torch.no_grad()
def evaluate(model, config, rank, out, step):
    """Full official Ref-SAV frames, same local DAVIS scorer as the baseline."""
    from PIL import Image
    from torch.nn import functional as F
    from local_scripts.evaluate_refsav_jf import merge_gt_mask, expression_anno_ids, binary_jaccard, boundary_f_measure
    root = Path(config["data_root"]) / "ref_sav_eval"
    meta = json.loads((root / "meta_expressions_valid.json").read_text(encoding="utf-8"))["videos"]
    masks = json.loads((root / "mask_dict.json").read_text(encoding="utf-8"))
    samples = [(v, e, info, item) for v, info in meta.items() for e, item in info["expressions"].items()]
    if len(samples) != 1945:
        raise ValueError("Full official validation must include 1945 expressions")
    state = rng_state()
    model.eval()
    rows = []
    for number in range(rank, len(samples), dist.get_world_size()):
        video, exp, info, item = samples[number]
        frames = []
        for name in sorted(info["frames"]):
            with Image.open(root / "videos" / video / f"{name}.jpg") as image:
                frames.append(image.convert("RGB"))
        pred, generation = model.predict(frames, item["exp"])
        js, fs = [], []
        for i, frame in enumerate(frames):
            gt = merge_gt_mask(masks, expression_anno_ids(item), i)
            if gt is None or gt.shape != (frame.height, frame.width):
                raise ValueError(f"Invalid GT frame alignment: {video}/{exp}/{i}")
            binary = np.zeros_like(gt) if pred is None else (
                F.interpolate(pred[i:i+1].float(), size=gt.shape, mode="bilinear", align_corners=False)[0, 0] > 0
            ).cpu().numpy()
            js.append(binary_jaccard(binary, gt))
            fs.append(boundary_f_measure(binary, gt, .008))
        j, f = float(np.mean(js)), float(np.mean(fs))
        rows.append({"video_id": video, "exp_id": str(exp), "type_id": int(item["type_id"]),
                     "J": j, "F": f, "J&F": (j+f)/2, **generation})
        if len(rows) % 20 == 0:
            print(f"SEG_VAL_PROGRESS rank={rank} step={step} expressions={len(rows)}", flush=True)
        del pred, frames
    rows = [r for shard in gather(rows) for r in shard]
    if len(rows) != 1945:
        raise ValueError("Incomplete distributed evaluation")
    groups = validation_summary(rows)
    if rank == 0:
        from focusseg_r1.full_runtime import atomic_json
        atomic_json(out / f"validation_{step:08d}.json", {
            "step": step, "groups": groups, "per_expression": rows,
            "generation_failures": sum(r["generation_failed"] for r in rows),
            "protocol": "official Ref-SAV val; local DAVIS-style scorer; full videos; first-five-frame prompts",
            "used_for_selection": True, "not_independent_test": True})
        print("SEG_FULL_VALIDATION " + json.dumps({"step": step, "groups": groups}), flush=True)
    model.train()
    restore_rng(state)
    return groups["PaperBalanced"]["J&F"]
