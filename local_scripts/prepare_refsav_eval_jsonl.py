from __future__ import annotations

"""Convert Ref-SAV Eval metadata to the local Sa2VA-Pro JSONL format.

This follows the official Sa2VA RefVOS evaluation data traversal:
one sample per expression, sorted expression ids, and all sorted video
frames from meta_expressions_valid.json. The ground-truth masks are decoded
only for metric computation; they are never used to choose input frames.
"""

import argparse
import json
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
DEFAULT_EVAL_ROOT = DEFAULT_DATA_ROOT / "ref_sav_eval"
DEFAULT_META = DEFAULT_EVAL_ROOT / "meta_expressions_valid.json"
DEFAULT_MASK_DICT = DEFAULT_EVAL_ROOT / "mask_dict.json"
DEFAULT_VIDEO_ROOT = DEFAULT_EVAL_ROOT / "videos"
DEFAULT_OUT_ROOT = Path("/XYFS02/HDD_POOL/lzu_bmwang/lzu_bmwang_1/data/sa2va_pro/processed/refsav_eval")


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


def safe_prepare_output(out_root: Path, overwrite: bool) -> tuple[Path, Path, Path]:
    if overwrite and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    mask_root = out_root / "masks"
    mask_root.mkdir(parents=True, exist_ok=True)
    out_json = out_root / "refsav_eval.jsonl"
    tmp_json = out_root / "refsav_eval.jsonl.tmp"
    if tmp_json.exists():
        tmp_json.unlink()
    return out_json, tmp_json, mask_root


def expression_text(expr: dict[str, Any]) -> str:
    value = expr.get("exp", "")
    return value.strip() if isinstance(value, str) else ""


def official_text_prompt(caption: str) -> str:
    if "?" in caption:
        return f"{caption} Please respond with a segmentation mask."
    return f"Please segment {caption}."


def anno_ids(expr: dict[str, Any]) -> list[str]:
    values = expr.get("anno_id", [])
    if not isinstance(values, list):
        values = [values]
    return [str(v) for v in values]


def obj_ids(expr: dict[str, Any]) -> list[int]:
    values = expr.get("obj_id", [])
    if not isinstance(values, list):
        values = [values]
    out: list[int] = []
    for value in values:
        try:
            out.append(int(value))
        except Exception:
            pass
    return out


def merged_mask_for_frame(
    mask_dict: dict[str, Any],
    anno_id_values: list[str],
    frame_index: int,
) -> np.ndarray | None:
    merged: np.ndarray | None = None
    for anno_id in anno_id_values:
        masks = mask_dict.get(anno_id)
        if not isinstance(masks, list) or frame_index >= len(masks):
            return None
        mask = decode_rle(masks[frame_index])
        if merged is None:
            merged = mask
        else:
            merged = np.maximum(merged, mask)
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description="Build official-aligned Ref-SAV Eval jsonl and PNG masks.")
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--mask-dict", type=Path, default=DEFAULT_MASK_DICT)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--no-overwrite", action="store_true", help="Keep existing output directory.")
    parser.add_argument("--postfix-every", type=int, default=100)
    args = parser.parse_args()

    eval_root = args.eval_root.resolve()
    meta_path = args.meta.resolve()
    mask_dict_path = args.mask_dict.resolve()
    video_root = args.video_root.resolve()
    out_root = args.out_root.resolve()
    overwrite = not args.no_overwrite

    for path in (eval_root, meta_path, mask_dict_path, video_root):
        if not path.exists():
            raise FileNotFoundError(path)

    out_json, tmp_json, mask_root = safe_prepare_output(out_root, overwrite=overwrite)
    meta = load_json(meta_path)
    mask_dict = load_json(mask_dict_path)
    videos = meta["videos"]
    total_expressions = sum(len(item.get("expressions", {})) for item in videos.values())
    stats = {
        "written": 0,
        "missing_video_dir": 0,
        "missing_frame_file": 0,
        "no_frames": 0,
        "empty_expression": 0,
        "missing_anno": 0,
        "mask_decode_failed": 0,
        "all_empty_masks": 0,
        "total_frames": 0,
        "total_masks": 0,
    }

    print(f"Eval root: {eval_root}")
    print(f"Videos: {len(videos)}")
    print(f"Expressions: {total_expressions}")
    print(f"Output root: {out_root}")
    print(f"Overwrite: {overwrite}")
    print("Progress: start")

    try:
        with tmp_json.open("w", encoding="utf-8") as fout:
            bar = tqdm(total=total_expressions, unit="expr", desc="Ref-SAV-Eval", dynamic_ncols=True) if tqdm else None
            for video_id, item in videos.items():
                expressions = item.get("expressions", {})
                frame_names = sorted(str(frame) for frame in item.get("frames", []) if str(frame).strip())
                frame_indices = list(range(len(frame_names)))
                video_dir = video_root / video_id
                if not video_dir.is_dir():
                    stats["missing_video_dir"] += len(expressions)
                    if bar:
                        bar.update(len(expressions))
                    continue
                if not frame_names:
                    stats["no_frames"] += len(expressions)
                    if bar:
                        bar.update(len(expressions))
                    continue

                missing_frames = [name for name in frame_names if not (video_dir / f"{name}.jpg").is_file()]
                if missing_frames:
                    stats["missing_frame_file"] += len(expressions)
                    if bar:
                        bar.update(len(expressions))
                    continue

                for exp_id in sorted(expressions.keys(), key=str):
                    expr = expressions[exp_id]
                    if bar:
                        bar.update(1)

                    caption = expression_text(expr)
                    if not caption:
                        stats["empty_expression"] += 1
                        continue

                    current_anno_ids = anno_ids(expr)
                    if not current_anno_ids or any(anno_id not in mask_dict for anno_id in current_anno_ids):
                        stats["missing_anno"] += 1
                        continue

                    masks: list[np.ndarray] = []
                    try:
                        for frame_index in frame_indices:
                            mask = merged_mask_for_frame(mask_dict, current_anno_ids, frame_index)
                            if mask is None:
                                raise ValueError(f"Missing mask for {video_id} exp={exp_id} frame_index={frame_index}")
                            masks.append(mask)
                    except Exception:
                        stats["mask_decode_failed"] += 1
                        continue

                    areas = [int((mask > 0).sum()) for mask in masks]
                    if sum(areas) <= 0:
                        stats["all_empty_masks"] += 1

                    mask_dir = mask_root / video_id / f"exp_{exp_id}"
                    mask_dir.mkdir(parents=True, exist_ok=True)
                    mask_paths = []
                    for frame_name, mask in zip(frame_names, masks):
                        mask_path = mask_dir / f"{frame_name}.png"
                        Image.fromarray(mask).save(mask_path)
                        mask_paths.append(str(mask_path))

                    problem = official_text_prompt(caption)
                    sample = {
                        "data_type": "video",
                        "path": str(video_dir),
                        "nframes": len(frame_names),
                        "frame_names": frame_names,
                        "frames": frame_names,
                        "problem_type": "segmentation",
                        "problem": problem,
                        "process": "",
                        "solution": "<answer>[SEG]</answer>",
                        "mask_paths": mask_paths,
                        "source_dataset": "ref_sav_eval",
                        "source_split": "valid",
                        "video_id": video_id,
                        "exp_id": str(exp_id),
                        "obj_id": obj_ids(expr),
                        "anno_id": current_anno_ids,
                        "metadata": {
                            "source": "Ref-SAV-Eval",
                            "height": item.get("height"),
                            "width": item.get("width"),
                            "type_id": expr.get("type_id"),
                            "frame_indices": frame_indices,
                            "frame_names": frame_names,
                            "mask_areas": areas,
                            "official_text_prompt": f"<image>\n{problem}",
                        },
                    }
                    fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    stats["written"] += 1
                    stats["total_frames"] += len(frame_names)
                    stats["total_masks"] += len(mask_paths)

                    if bar and args.postfix_every > 0 and stats["written"] % args.postfix_every == 0:
                        skipped = (
                            stats["missing_video_dir"]
                            + stats["missing_frame_file"]
                            + stats["no_frames"]
                            + stats["empty_expression"]
                            + stats["missing_anno"]
                            + stats["mask_decode_failed"]
                        )
                        bar.set_postfix(written=stats["written"], masks=stats["total_masks"], skipped=skipped)

            if bar:
                bar.close()

        tmp_json.replace(out_json)
    except KeyboardInterrupt:
        if tmp_json.exists():
            tmp_json.unlink()
        raise

    skipped = {
        key: stats[key]
        for key in ("missing_video_dir", "missing_frame_file", "no_frames", "empty_expression", "missing_anno", "mask_decode_failed")
        if stats[key]
    }
    diagnostics = {
        key: stats[key]
        for key in ("all_empty_masks", "total_frames", "total_masks")
        if stats[key]
    }
    print(f"written: {stats['written']}")
    print(f"skipped: {skipped}")
    print(f"diagnostics: {diagnostics}")
    print(f"out_json: {out_json}")
    print(f"mask_root: {mask_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
