"""Fresh seven-source SEG LoRA: prepare -> four-GPU smoke -> three epochs."""
import argparse
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random
import time
from seg_lora.data import CompleteIndexBuilder, read_config, open_store
from focusseg_r1.full_data import file_digest, config_digest, stable_seed
from focusseg_r1.full_runtime import atomic_json, epoch_plan, lr_multiplier
from local_scripts.seg_lora_storage import (
    atomic_checkpoint, compatible_provenance, keep_latest_best, recover_best,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["prepare", "smoke", "train"], required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke-report", type=Path)
    parser.add_argument("--skip-smoke", action="store_true",
                        help="Explicit user-authorized direct training; retain all index/runtime checks")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--storage-upgrade-reference", type=Path,
                        help="Original run.json copy: allow only the reviewed checkpoint storage upgrade")
    args = parser.parse_args()
    if args.storage_upgrade_reference is not None and args.resume is None:
        parser.error("--storage-upgrade-reference requires --resume")
    if args.skip_smoke and (args.mode != "train" or args.smoke_report is not None):
        parser.error("--skip-smoke requires train mode and cannot accompany --smoke-report")
    config = read_config(args.config)
    if args.mode == "prepare":
        report = CompleteIndexBuilder(config, args.index).build()
        args.out.mkdir(parents=True, exist_ok=False)
        atomic_json(args.out / "index_report.json", report)
        store = open_store(args.index, config)
        checks = []
        for sample_id in store.fixed_ids("train", 3, config["seed"]):
            sample = store.load(sample_id, stable_seed(config["seed"], sample_id), config["frames"])
            checks.append({"uid": sample["uid"], "positive_pixels": int(sample["masks"].sum())})
        row = store.db.execute("SELECT id FROM samples WHERE source='revos' AND mask_spec LIKE '%\"ids\": []%' LIMIT 1").fetchone()
        if row is None:
            raise ValueError("No explicit negative sample found")
        empty = store.load(row[0], config["seed"], config["frames"])
        if empty["masks"].any():
            raise ValueError("Empty target decoded to nonempty mask")
        atomic_json(args.index.with_suffix(".ready.json"), {"status": "COMPLETE_SEVEN_SOURCE_INDEX",
                    "index_sha256": file_digest(args.index), "config_digest": config_digest(config),
                    "sampled_decodes": checks, "empty_target_uid": empty["uid"]})
        print("SEG_INDEX_READY", flush=True)
        return

    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from seg_lora.model import SegLoRA
    from seg_lora.runtime import identity, rng_state, restore_rng, gather, evaluate
    rank, world, local = (int(os.environ[k]) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    if world != 4 or "SLURM_JOB_ID" not in os.environ:
        raise ValueError("Run inside a four-GPU Slurm allocation")
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", timeout=timedelta(hours=12))
    provenance = {"config_digest": config_digest(config), "code": identity()}
    provenance["code"]["local_scripts/seg_lora_storage.py"] = file_digest(
        Path(__file__).with_name("seg_lora_storage.py"))
    storage_reference = None
    if args.storage_upgrade_reference:
        storage_reference = json.loads(args.storage_upgrade_reference.read_text(encoding="utf-8"))["provenance"]
    digest = [file_digest(args.index) if rank == 0 else None]
    dist.broadcast_object_list(digest, src=0)
    provenance["index_sha256"] = digest[0]
    ready = json.loads(args.index.with_suffix(".ready.json").read_text(encoding="utf-8"))
    if ready["index_sha256"] != digest[0] or ready["config_digest"] != provenance["config_digest"]:
        raise ValueError("Index readiness/provenance mismatch")
    if args.mode == "train" and not args.skip_smoke:
        if args.smoke_report is None:
            raise ValueError("Four-GPU smoke report is required before long training")
        smoke = json.loads(args.smoke_report.read_text(encoding="utf-8"))
        if smoke.get("status") != "FOUR_GPU_SEG_SMOKE_OK" or not compatible_provenance(
                smoke.get("provenance"), provenance, storage_reference):
            raise ValueError("Smoke report absent or belongs to another code/config/index")
    if args.skip_smoke and rank == 0:
        print("SMOKE_SKIPPED_BY_EXPLICIT_USER_REQUEST: first real four-GPU model execution", flush=True)
    if rank == 0:
        args.out.mkdir(parents=True, exist_ok=bool(args.resume))
    dist.barrier()
    random.seed(config["seed"] + rank)
    np.random.seed(config["seed"] + rank)
    torch.manual_seed(config["seed"] + rank)
    torch.cuda.manual_seed_all(config["seed"] + rank)
    store = open_store(args.index, config)
    ids = [row[0] for row in store.ids("train")]
    batch = world * config["gradient_accumulation"]
    per_epoch = math.ceil(len(ids) / batch)
    total = per_epoch * config["epochs"]
    model = SegLoRA(config).to(local).train()
    qa_input = None
    qa_before = None
    mask_gradient_norm = None
    if args.mode == "smoke":
        sample = store.load(ids[rank], config["seed"] + rank, 5)
        result = model(sample["images"], sample["text"], sample["masks"])
        lora_b = [p for n, p in model.named_parameters() if "lora_B" in n and p.requires_grad]
        gradients = torch.autograd.grad(result["mask_loss"], lora_b, allow_unused=True)
        mask_gradient_norm = sum(float(g.detach().float().abs().sum()) for g in gradients if g is not None)
        if not math.isfinite(mask_gradient_norm) or mask_gradient_norm <= 0:
            raise RuntimeError("Mask-only supervision does not reach LoRA")
        del result, gradients
        qa_input = model.processor.tokenizer("Describe a panda.", return_tensors="pt").to(local)
        with model.qa_route() as backbone, torch.no_grad():
            qa_before = backbone(**qa_input, use_cache=False).logits.detach().cpu()
    ddp = DDP(model, device_ids=[local], broadcast_buffers=False, find_unused_parameters=False)
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in trainable if not n.startswith("base.text_hidden_fcs.")], "lr": config["learning_rate"]},
        {"params": [p for n, p in trainable if n.startswith("base.text_hidden_fcs.")], "lr": config["projection_learning_rate"]}],
        betas=(.9, .999), weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_multiplier(step, total, config["warmup_ratio"]))
    step, best = 0, None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if checkpoint.get("format") != "seg_lora_fresh_v1" or not compatible_provenance(
                checkpoint["provenance"], provenance, storage_reference):
            raise ValueError("Resume rejects historical query or mismatched provenance")
        model.load_trainable(checkpoint["trainable"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        step, best = checkpoint["step"], checkpoint["best"]
        best = recover_best(args.out, best)
        restore_rng(checkpoint["rng"][rank])
        del checkpoint
    if rank == 0:
        atomic_json(args.out / "run.json", {"config": config, "provenance": provenance,
                    "smoke_skipped_by_user": args.skip_smoke,
                    "trainable_parameters": sum(p.numel() for _, p in trainable),
                    "trainable_names": [n for n, _ in trainable], "lora_targets": model.lora_targets,
                    "train_expressions": len(ids), "steps_per_epoch": per_epoch, "total_steps": total,
                    "world_size": world, "effective_batch": batch, "fresh": args.resume is None})
        atomic_json(args.out / f"storage_policy_{os.environ['SLURM_JOB_ID']}.json", {
            "policy": "latest_and_best", "resume": str(args.resume) if args.resume else None,
            "storage_upgrade_reference": str(args.storage_upgrade_reference) if args.storage_upgrade_reference else None,
            "original_provenance": storage_reference, "current_provenance": provenance,
            "step_at_start": step})
        print(f"SEG_START mode={args.mode} samples={len(ids)} steps_per_epoch={per_epoch} total={total}", flush=True)

    saved_step = None
    saved_digest = None

    def save(score=None):
        nonlocal best, saved_step, saved_digest
        states = gather(rng_state())
        if score is not None and (best is None or score > best["score"]):
            best = {"score": score, "step": step, "file": f"checkpoint_{step:08d}.pt"}
        if rank == 0:
            path = args.out / f"checkpoint_{step:08d}.pt"
            # Validation restores RNG and does not update weights/optimizer. Its
            # second save only changes selection metadata, so do not rewrite 4GB.
            if saved_step != step:
                atomic_checkpoint(path, {"format": "seg_lora_fresh_v1", "step": step, "config": config,
                            "provenance": provenance, "trainable": model.trainable_state(),
                            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                            "rng": states, "best": best}, torch.save)
                saved_digest = file_digest(path)
                saved_step = step
            atomic_json(args.out / "latest.json", {"file": path.name, "step": step, "sha256": saved_digest})
            if best:
                atomic_json(args.out / "best.json", best)
            manifest_path = args.out / "checkpoints.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else []
            previous = next((r for r in manifest if r["step"] == step), {})
            manifest = [r for r in manifest if r["step"] != step]
            manifest.append({"step": step, "file": path.name,
                             "score": score if score is not None else previous.get("score"), "retained": True})
            deleted = keep_latest_best(args.out, path.name, best["file"] if best else None)
            for record in manifest:
                record["retained"] = (args.out / record["file"]).is_file()
            atomic_json(manifest_path, manifest)
            print("SEG_CHECKPOINT_SAVED " + json.dumps({"step": step, "latest": path.name,
                  "best": best["file"] if best else None, "deleted": deleted,
                  "policy": "latest_and_best"}), flush=True)
        dist.barrier()

    if args.mode == "smoke":
        smoke_ids = store.fixed_ids("train", 1, config["seed"])
        negative = store.db.execute("SELECT id FROM samples WHERE source='revos' AND mask_spec LIKE '%\"ids\": []%' LIMIT 1").fetchone()[0]
        smoke_ids.append(negative)
        records = []
        # Two updates cover all seven sources plus one explicit negative, across four ranks.
        for i in range(2):
            sample = store.load(smoke_ids[i * 4 + rank], stable_seed(config["seed"], i, rank), 5)
            optimizer.zero_grad(set_to_none=True)
            result = ddp(sample["images"], sample["text"], sample["masks"])
            result["loss"].backward()
            if not torch.isfinite(result["loss"]) or not any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in trainable if "lora_B" in n):
                raise RuntimeError("LoRA has no finite training signal")
            if any(p.grad is None or not torch.isfinite(p.grad).all() for _, p in trainable):
                raise RuntimeError("Missing/nonfinite trainable gradient")
            torch.nn.utils.clip_grad_norm_([p for _, p in trainable], config["max_grad_norm"])
            optimizer.step()
            records.append({"uid": sample["uid"], "loss": float(result["loss"].detach()),
                            "ce": float(result["ce"]), "bce": float(result["bce"]), "dice": float(result["dice"])})
        step = 2
        save()
        checkpoint = torch.load(args.out / "checkpoint_00000002.pt", map_location="cpu", weights_only=True)
        before = model.trainable_state()
        model.load_trainable(checkpoint["trainable"])
        if any(not torch.equal(before[k], checkpoint["trainable"][k]) for k in before):
            raise RuntimeError("Distributed saved parameters differ")
        model.eval()
        with model.qa_route() as backbone, torch.no_grad():
            qa_after = backbone(**qa_input, use_cache=False).logits.detach().cpu()
        if not torch.equal(qa_before, qa_after):
            raise RuntimeError("Disabled-adapter QA logits changed after segmentation updates")
        _, generation = model.predict(sample["images"], sample["text"])
        reports = gather({"rank": rank, "samples": records, "generation": generation,
                          "mask_only_lora_gradient_l1": mask_gradient_norm, "qa_logits_unchanged": True,
                          "peak_gpu_gib": torch.cuda.max_memory_allocated() / 1024**3})
        if rank == 0:
            atomic_json(args.out / "smoke.json", {"status": "FOUR_GPU_SEG_SMOKE_OK", "provenance": provenance,
                        "ranks": reports, "generation_quality_not_pass_criterion": True,
                        "note": "Disposable two-update smoke weights; train mode starts from original weights."})
        dist.destroy_process_group()
        return

    if step == 0:
        save()
    start = time.monotonic()
    from contextlib import nullcontext
    while step < total:
        epoch, offset = divmod(step, per_epoch)
        plan, padding = epoch_plan(ids, config["seed"], epoch, batch)
        for update in range(offset, per_epoch):
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            chunk = plan[update * batch:(update + 1) * batch][rank::world]
            for micro, sample_id in enumerate(chunk):
                sample = store.load(sample_id, stable_seed(config["seed"], epoch, update, rank, micro), 5)
                sync = nullcontext() if micro == len(chunk) - 1 else ddp.no_sync()
                with sync:
                    result = ddp(sample["images"], sample["text"], sample["masks"])
                    if not torch.isfinite(result["loss"]):
                        raise FloatingPointError("Nonfinite supervised loss")
                    (result["loss"] / len(chunk)).backward()
                    loss_sum += float(result["loss"].detach()) / len(chunk)
            norm = torch.nn.utils.clip_grad_norm_([p for _, p in trainable], config["max_grad_norm"], error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            step += 1
            if rank == 0 and step % 10 == 0:
                print("SEG_TRAIN_STEP " + json.dumps({"step": step, "epoch_fraction": step / per_epoch,
                      "rank0_loss": loss_sum, "grad_norm": float(norm), "lr_next": scheduler.get_last_lr(),
                      "seconds": time.monotonic() - start, "peak_gpu_gib": torch.cuda.max_memory_allocated()/1024**3}), flush=True)
            do_eval = step % config["validation_every"] == 0 or step % per_epoch == 0
            if step % config["save_every"] == 0 or do_eval:
                save()
            if do_eval:
                score = evaluate(model, config, rank, args.out, step)
                save(score)
            if step == total:
                break
    if rank == 0:
        atomic_json(args.out / "finished.json", {"status": "THREE_EPOCHS_COMPLETE", "step": step, "best": best})
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
