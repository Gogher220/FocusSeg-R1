from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

try:
    import pycocotools.mask as mask_utils
except Exception as exc:  # pragma: no cover
    raise SystemExit("pycocotools is required. Install it in the active env first.") from exc


DEFAULT_DATA_ROOT = Path("/XYFS02/HDD_POOL/lzu_bmwang/lzu_bmwang_1/data/Sa2VA-data")
DEFAULT_SAV_ROOT = DEFAULT_DATA_ROOT / "video_datas" / "sam_v_full"
DEFAULT_REF_SAV = DEFAULT_DATA_ROOT / "video_datas" / "Ref-SAV.json"
DEFAULT_OUT_ROOT = Path("/XYFS02/HDD_POOL/lzu_bmwang/lzu_bmwang_1/data/sa2va_pro/processed/refsav_sft_full")
DEFAULT_OUT_JSON = DEFAULT_OUT_ROOT / "refsav_sft_full.jsonl"
DEFAULT_MASK_ROOT = DEFAULT_OUT_ROOT / "masks"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def decode_rle(rle: dict[str, Any]) -> np.ndarray:
    item = dict(rle)
    counts = item.get("counts")
    if isinstance(counts, str):
        item["counts"] = counts.encode("utf-8")
    mask = mask_utils.decode(item)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return ((mask > 0).astype(np.uint8)) * 255


def caption_of(obj: dict[str, Any]) -> str:
    for key in ("formated", "final_caption", "video_caption", "image_caption", "crop_category"):
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def build_indices(sav_root: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    train_root = sav_root / "sav_train"
    video_index = {p.stem: p for p in train_root.glob("*/*.mp4")}
    anno_index = {p.name.replace("_manual.json", ""): p for p in train_root.glob("*/*_manual.json")}
    return video_index, anno_index


def safe_prepare_output(out_root: Path, overwrite: bool) -> tuple[Path, Path, Path]:
    if overwrite and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    mask_root = out_root / "masks"
    mask_root.mkdir(parents=True, exist_ok=True)
    out_json = out_root / "refsav_sft_full.jsonl"
    tmp_json = out_root / "refsav_sft_full.jsonl.tmp"
    if tmp_json.exists():
        tmp_json.unlink()
    return out_json, tmp_json, mask_root


def main() -> int:
    parser = argparse.ArgumentParser(description="Build full Ref-SAV SFT jsonl and PNG masks.")
    parser.add_argument("--ref-sav", type=Path, default=DEFAULT_REF_SAV)
    parser.add_argument("--sav-root", type=Path, default=DEFAULT_SAV_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--nframes", type=int, default=5)
    parser.add_argument("--no-overwrite", action="store_true", help="Keep existing output directory.")
    parser.add_argument("--postfix-every", type=int, default=200, help="Update tqdm postfix every N written samples.")
    args = parser.parse_args()

    overwrite = not args.no_overwrite
    ref_sav = args.ref_sav.resolve()
    sav_root = args.sav_root.resolve()
    out_root = args.out_root.resolve()

    if not ref_sav.exists():
        raise FileNotFoundError(f"Missing Ref-SAV json: {ref_sav}")
    if not sav_root.exists():
        raise FileNotFoundError(f"Missing SAM-V root: {sav_root}")

    out_json, tmp_json, mask_root = safe_prepare_output(out_root, overwrite=overwrite)
    ref_data = load_json(ref_sav)
    video_index, anno_index = build_indices(sav_root)

    total_objects = sum(len(item.get("objects", {})) for item in ref_data.values())
    stats = {
        "written": 0,
        "missing_video_or_anno": 0,
        "bad_anno": 0,
        "empty_caption": 0,
        "bad_obj_id": 0,
        "obj_not_in_masklet": 0,
        "decode_failed": 0,
        "all_empty_masks": 0,
    }

    print(f"Ref-SAV videos: {len(ref_data)}")
    print(f"Video index: {len(video_index)}")
    print(f"Anno index: {len(anno_index)}")
    print(f"Total object candidates: {total_objects}")
    print(f"Output root: {out_root}")
    print(f"Overwrite: {overwrite}")
    print("Progress: start")

    try:
        with tmp_json.open("w", encoding="utf-8") as fout:
            if tqdm is not None:
                bar = tqdm(total=total_objects, unit="obj", desc="Ref-SAV", dynamic_ncols=True)
            else:  # pragma: no cover
                bar = None

            processed = 0
            for vid_key, item in ref_data.items():
                video_id = item.get("video_id", vid_key)
                objects = item.get("objects", {})
                video_path = video_index.get(video_id)
                anno_path = anno_index.get(video_id)

                if video_path is None or anno_path is None:
                    stats["missing_video_or_anno"] += len(objects)
                    processed += len(objects)
                    if bar is not None:
                        bar.update(len(objects))
                        bar.set_postfix(written=stats["written"], skipped=processed - stats["written"])
                    continue

                try:
                    ann = load_json(anno_path)
                    masklet = ann["masklet"]
                    masklet_ids = [int(x) for x in ann["masklet_id"]]
                    id_to_pos = {obj_id: i for i, obj_id in enumerate(masklet_ids)}
                except Exception:
                    stats["bad_anno"] += len(objects)
                    processed += len(objects)
                    if bar is not None:
                        bar.update(len(objects))
                        bar.set_postfix(written=stats["written"], skipped=processed - stats["written"])
                    continue

                if not masklet or not masklet_ids:
                    stats["bad_anno"] += len(objects)
                    processed += len(objects)
                    if bar is not None:
                        bar.update(len(objects))
                        bar.set_postfix(written=stats["written"], skipped=processed - stats["written"])
                    continue

                frame_ids = np.linspace(0, len(masklet) - 1, args.nframes, dtype=int).tolist()

                for obj_key, obj in objects.items():
                    processed += 1
                    if bar is not None:
                        bar.update(1)

                    caption = caption_of(obj)
                    if not caption:
                        stats["empty_caption"] += 1
                        continue

                    try:
                        obj_id = int(obj.get("obj_id", obj_key))
                    except Exception:
                        stats["bad_obj_id"] += 1
                        continue

                    pos = id_to_pos.get(obj_id)
                    if pos is None:
                        stats["obj_not_in_masklet"] += 1
                        continue

                    try:
                        masks = [decode_rle(masklet[fi][pos]) for fi in frame_ids]
                        areas = [int((mask > 0).sum()) for mask in masks]
                    except Exception:
                        stats["decode_failed"] += 1
                        continue

                    if sum(areas) <= 0:
                        stats["all_empty_masks"] += 1
                        continue

                    mask_dir = mask_root / video_id / f"obj_{obj_id}"
                    mask_dir.mkdir(parents=True, exist_ok=True)
                    mask_paths = []
                    for local_i, mask in enumerate(masks):
                        mask_path = mask_dir / f"frame_{local_i:02d}.png"
                        Image.fromarray(mask).save(mask_path)
                        mask_paths.append(str(mask_path))

                    sample = {
                        "data_type": "video",
                        "video_path": str(video_path),
                        "nframes": args.nframes,
                        "problem_type": "segmentation",
                        "problem": f"Please segment the object described as follows in this video: {caption}",
                        "process": "",
                        "solution": f"<answer>{caption} [SEG]</answer>",
                        "target_masks": mask_paths,
                        "metadata": {
                            "source": "Ref-SAV",
                            "video_id": video_id,
                            "obj_id": obj_id,
                            "anno_path": str(anno_path),
                            "frame_ids_in_masklet": frame_ids,
                            "mask_areas": areas,
                        },
                    }
                    fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    stats["written"] += 1

                    if args.postfix_every > 0 and stats["written"] % args.postfix_every == 0:
                        msg = f"Progress written={stats['written']} skipped={processed - stats['written']} current_video={video_id}"
                        if bar is not None:
                            bar.set_postfix(written=stats["written"], skipped=processed - stats["written"])
                        else:  # pragma: no cover
                            print(msg, flush=True)

                if bar is None and processed % max(args.postfix_every, 1) == 0:  # pragma: no cover
                    print(f"Progress processed={processed} written={stats['written']} skipped={processed - stats['written']}", flush=True)

            if bar is not None:
                bar.close()

        tmp_json.replace(out_json)
    except KeyboardInterrupt:
        print("\nInterrupted. Partial temp output left at:", tmp_json)
        print("Rerun the same script with the default overwrite behavior to clear bad content and start over.")
        return 130

    print("written:", stats["written"])
    print("skipped:", {k: v for k, v in stats.items() if k != "written"})
    print("out_json:", out_json)
    print("mask_root:", mask_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
