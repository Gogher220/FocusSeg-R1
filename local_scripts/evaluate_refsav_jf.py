from __future__ import annotations

"""Evaluate Ref-SAV predictions with video segmentation J/F/J&F metrics.

The official Sa2VA Ref-SAV eval script emits per-frame masks but does not
include the final scorer. This script fills that gap for local experiments:
it reads the official Ref-SAV metadata/GT masks, reads predicted masks, and
reports Jaccard IoU (J), boundary F-measure (F), and their average (J&F).
"""

import argparse
import json
import math
from collections import defaultdict
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

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

try:
    from scipy import ndimage  # type: ignore
except Exception:  # pragma: no cover
    ndimage = None


DEFAULT_DATA_ROOT = Path("/XYFS02/HDD_POOL/lzu_bmwang/lzu_bmwang_1/data/Sa2VA-data")
DEFAULT_EVAL_ROOT = DEFAULT_DATA_ROOT / "ref_sav_eval"
DEFAULT_META = DEFAULT_EVAL_ROOT / "meta_expressions_valid.json"
DEFAULT_MASK_DICT = DEFAULT_EVAL_ROOT / "mask_dict.json"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_rle(rle: dict[str, Any]) -> dict[str, Any]:
    item = dict(rle)
    counts = item.get("counts")
    if isinstance(counts, str):
        item["counts"] = counts.encode("utf-8")
    return item


def decode_rle(rle: dict[str, Any]) -> np.ndarray:
    mask = mask_utils.decode(normalize_rle(rle))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask.astype(bool)


def merge_gt_mask(mask_dict: dict[str, Any], anno_ids: list[str], frame_index: int) -> np.ndarray | None:
    merged: np.ndarray | None = None
    for anno_id in anno_ids:
        frames = mask_dict.get(str(anno_id))
        if not isinstance(frames, list) or frame_index >= len(frames):
            return None
        mask = decode_rle(frames[frame_index])
        merged = mask if merged is None else np.logical_or(merged, mask)
    return merged


def expression_anno_ids(expr: dict[str, Any]) -> list[str]:
    anno_ids = expr.get("anno_id", [])
    if not isinstance(anno_ids, list):
        anno_ids = [anno_ids]
    return [str(anno_id) for anno_id in anno_ids]


def parse_type_labels(value: str) -> dict[int, str]:
    labels: dict[int, str] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        key, _, label = item.partition(":")
        if not key or not label:
            raise ValueError(f"Bad --type-id-labels item: {item!r}")
        labels[int(key)] = label
    return labels


def expression_type_label(expr: dict[str, Any], type_labels: dict[int, str]) -> str:
    value = expr.get("type_id")
    try:
        type_id = int(value)
    except Exception:
        return "Unknown"
    return type_labels.get(type_id, f"Type{type_id}")


def load_mask_image(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr > 0


def resize_bool_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask.astype(bool)
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    image = image.resize((shape[1], shape[0]), resample=Image.NEAREST)
    return np.asarray(image) > 0


def prediction_png_candidates(pred_root: Path, video_id: str, exp_id: str, frame_name: str) -> list[Path]:
    return [
        pred_root / "Annotations" / video_id / exp_id / f"{frame_name}.png",
        pred_root / "Annotations" / video_id / f"exp_{exp_id}" / f"{frame_name}.png",
        pred_root / video_id / exp_id / f"{frame_name}.png",
        pred_root / video_id / f"exp_{exp_id}" / f"{frame_name}.png",
        pred_root / "masks" / video_id / exp_id / f"{frame_name}.png",
        pred_root / "masks" / video_id / f"exp_{exp_id}" / f"{frame_name}.png",
    ]


def load_prediction_from_pngs(
    pred_root: Path,
    video_id: str,
    exp_id: str,
    frame_name: str,
) -> np.ndarray | None:
    for path in prediction_png_candidates(pred_root, video_id, exp_id, frame_name):
        if path.is_file():
            return load_mask_image(path)
    return None


def load_prediction_from_json(
    predictions: dict[str, Any],
    video_id: str,
    exp_id: str,
    frame_index: int,
    frame_name: str | None = None,
) -> np.ndarray | None:
    video_item = predictions.get(video_id)
    if not isinstance(video_item, dict):
        return None
    exp_item = video_item.get(str(exp_id))
    if isinstance(exp_item, dict):
        # The official Sa2VA results.json stores metadata and masks together.
        # Older local predictions stored the RLE list directly.
        frames = exp_item.get("frames")
        masks = exp_item.get("prediction_masks")
        if not isinstance(frames, list) or not isinstance(masks, list):
            raise ValueError(f"Malformed prediction for {video_id}/{exp_id}")
        if len(frames) != len(masks):
            raise ValueError(f"Frame/mask count differs for {video_id}/{exp_id}")
        if frame_name is not None and frame_index < len(frames):
            if str(frames[frame_index]) != str(frame_name):
                raise ValueError(
                    f"Frame order differs for {video_id}/{exp_id} at {frame_index}: "
                    f"{frames[frame_index]!r} != {frame_name!r}"
                )
        exp_item = masks
    if not isinstance(exp_item, list) or frame_index >= len(exp_item):
        return None
    return decode_rle(exp_item[frame_index])


def binary_jaccard(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    intersection = np.logical_and(pred, gt).sum()
    return float(intersection / union)


def mask_to_boundary(mask: np.ndarray) -> np.ndarray:
    """DAVIS-style one-pixel boundary map (the seg2bmap construction)."""
    mask = mask.astype(bool)
    east = np.zeros_like(mask)
    south = np.zeros_like(mask)
    southeast = np.zeros_like(mask)
    east[:, :-1] = mask[:, 1:]
    south[:-1, :] = mask[1:, :]
    southeast[:-1, :-1] = mask[1:, 1:]
    boundary = (mask ^ east) | (mask ^ south) | (mask ^ southeast)
    boundary[-1, :] = mask[-1, :] ^ east[-1, :]
    boundary[:, -1] = mask[:, -1] ^ south[:, -1]
    boundary[-1, -1] = False
    return boundary


def disk_kernel(radius: int) -> np.ndarray:
    radius = max(1, int(radius))
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (yy * yy + xx * xx <= radius * radius).astype(np.uint8)


def dilate_boundary(boundary: np.ndarray, radius: int) -> np.ndarray:
    if not boundary.any():
        return boundary.astype(bool)
    kernel = disk_kernel(radius)
    if cv2 is not None:
        return cv2.dilate(boundary.astype(np.uint8), kernel, iterations=1).astype(bool)
    if ndimage is not None:
        return ndimage.binary_dilation(boundary, structure=kernel.astype(bool))

    dilated = boundary.astype(bool)
    for _ in range(max(1, radius)):
        padded = np.pad(dilated, 1, mode="constant", constant_values=False)
        next_mask = np.zeros_like(dilated, dtype=bool)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                next_mask |= padded[1 + dy : 1 + dy + dilated.shape[0], 1 + dx : 1 + dx + dilated.shape[1]]
        dilated = next_mask
    return dilated


def boundary_f_measure(pred: np.ndarray, gt: np.ndarray, bound_th: float = 0.008) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if not pred.any() and not gt.any():
        return 1.0
    if not pred.any() or not gt.any():
        return 0.0

    height, width = gt.shape
    radius = int(math.ceil(bound_th * math.sqrt(height * height + width * width)))
    pred_boundary = mask_to_boundary(pred)
    gt_boundary = mask_to_boundary(gt)
    if not pred_boundary.any() and not gt_boundary.any():
        return 1.0
    if not pred_boundary.any() or not gt_boundary.any():
        return 0.0

    pred_match = np.logical_and(pred_boundary, dilate_boundary(gt_boundary, radius)).sum()
    gt_match = np.logical_and(gt_boundary, dilate_boundary(pred_boundary, radius)).sum()
    precision = pred_match / max(1, pred_boundary.sum())
    recall = gt_match / max(1, gt_boundary.sum())
    if precision + recall == 0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def aggregate(samples: list[dict[str, Any]]) -> dict[str, float | int]:
    if not samples:
        return {"num_expressions": 0, "J": 0.0, "F": 0.0, "J&F": 0.0}
    mean_j = float(np.mean([item["J"] for item in samples]))
    mean_f = float(np.mean([item["F"] for item in samples]))
    return {
        "num_expressions": len(samples),
        "J": mean_j,
        "F": mean_f,
        "J&F": (mean_j + mean_f) / 2.0,
    }


def print_table(group_metrics: dict[str, dict[str, float | int]]) -> None:
    print("")
    print(f"{'Split':<10} {'N':>8} {'J':>10} {'F':>10} {'J&F':>10}")
    print("-" * 52)
    for name in ("Overall", "PaperBalanced", "Long", "Short", "Unknown"):
        if name not in group_metrics:
            continue
        item = group_metrics[name]
        print(
            f"{name:<10} {int(item['num_expressions']):>8} "
            f"{float(item['J']) * 100:>10.2f} "
            f"{float(item['F']) * 100:>10.2f} "
            f"{float(item['J&F']) * 100:>10.2f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Ref-SAV predictions with J/F/J&F.")
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--mask-dict", type=Path, default=DEFAULT_MASK_DICT)
    parser.add_argument("--pred-json", type=Path, default=None, help="Official results.json with RLE prediction masks.")
    parser.add_argument("--pred-root", type=Path, default=None, help="Prediction PNG root, e.g. work_dir/Annotations.")
    parser.add_argument("--allow-missing-predictions", action="store_true", help="Treat missing predictions as empty masks instead of failing.")
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--details-jsonl", type=Path, default=None)
    parser.add_argument("--boundary-th", type=float, default=0.008)
    parser.add_argument("--type-id-labels", default="0:Long,1:Short")
    parser.add_argument("--postfix-every", type=int, default=50)
    args = parser.parse_args()

    if args.pred_json is None and args.pred_root is None:
        raise ValueError("Provide --pred-json or --pred-root.")

    meta_path = args.meta.resolve()
    mask_dict_path = args.mask_dict.resolve()
    pred_json_path = args.pred_json.resolve() if args.pred_json else None
    pred_root = args.pred_root.resolve() if args.pred_root else None

    meta = load_json(meta_path)
    mask_dict = load_json(mask_dict_path)
    predictions = load_json(pred_json_path) if pred_json_path else None
    type_labels = parse_type_labels(args.type_id_labels)
    videos = meta["videos"]
    total_expressions = sum(len(item.get("expressions", {})) for item in videos.values())

    if args.out_json is None:
        if pred_json_path:
            out_json = pred_json_path.parent / "refsav_jf_metrics.json"
        elif pred_root:
            out_json = pred_root / "refsav_jf_metrics.json"
        else:
            out_json = Path("refsav_jf_metrics.json").resolve()
    else:
        out_json = args.out_json.resolve()

    stats = defaultdict(int)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    per_sample: list[dict[str, Any]] = []

    print(f"Meta: {meta_path}")
    print(f"Mask dict: {mask_dict_path}")
    print(f"Pred json: {pred_json_path}")
    print(f"Pred root: {pred_root}")
    print(f"Expressions: {total_expressions}")
    print("Progress: start")

    if args.details_jsonl:
        args.details_jsonl.parent.mkdir(parents=True, exist_ok=True)
    details_file = args.details_jsonl.open("w", encoding="utf-8") if args.details_jsonl else None
    try:
        bar = tqdm(total=total_expressions, unit="expr", desc="Ref-SAV-JF", dynamic_ncols=True) if tqdm else None
        for video_id, video_item in videos.items():
            frame_names = sorted(str(frame) for frame in video_item.get("frames", []) if str(frame).strip())
            expressions = video_item.get("expressions", {})
            for exp_id in sorted(expressions.keys(), key=str):
                expr = expressions[exp_id]
                if bar:
                    bar.update(1)

                anno_ids = expression_anno_ids(expr)
                if not anno_ids:
                    stats["missing_anno_id"] += 1
                    continue

                frame_j: list[float] = []
                frame_f: list[float] = []
                frame_count = 0
                for frame_index, frame_name in enumerate(frame_names):
                    gt = merge_gt_mask(mask_dict, anno_ids, frame_index)
                    if gt is None:
                        stats["missing_gt_frame"] += 1
                        continue

                    if predictions is not None:
                        pred = load_prediction_from_json(
                            predictions, video_id, str(exp_id), frame_index, frame_name
                        )
                    elif pred_root is not None:
                        pred = load_prediction_from_pngs(pred_root, video_id, str(exp_id), frame_name)
                    else:
                        pred = None

                    if pred is None:
                        stats["missing_pred_frame"] += 1
                        if predictions is not None and not args.allow_missing_predictions:
                            raise ValueError(
                                f"Missing prediction for {video_id}/{exp_id}/{frame_name}; "
                                "use --allow-missing-predictions only for an intentional sparse evaluation"
                            )
                        pred = np.zeros_like(gt, dtype=bool)
                    elif pred.shape != gt.shape:
                        stats["resized_pred_frame"] += 1
                        pred = resize_bool_mask(pred, gt.shape)

                    frame_j.append(binary_jaccard(pred, gt))
                    frame_f.append(boundary_f_measure(pred, gt, bound_th=args.boundary_th))
                    frame_count += 1

                if frame_count == 0:
                    stats["empty_expression_eval"] += 1
                    continue

                sample = {
                    "video_id": video_id,
                    "exp_id": str(exp_id),
                    "type_id": expr.get("type_id"),
                    "type_label": expression_type_label(expr, type_labels),
                    "num_frames": frame_count,
                    "J": float(np.mean(frame_j)),
                    "F": float(np.mean(frame_f)),
                }
                sample["J&F"] = (sample["J"] + sample["F"]) / 2.0
                per_sample.append(sample)
                groups["Overall"].append(sample)
                groups[sample["type_label"]].append(sample)
                if details_file:
                    details_file.write(json.dumps(sample, ensure_ascii=False) + "\n")

                if bar and args.postfix_every > 0 and len(per_sample) % args.postfix_every == 0:
                    bar.set_postfix(done=len(per_sample), missing_pred=stats["missing_pred_frame"])

        if bar:
            bar.close()
    finally:
        if details_file:
            details_file.close()

    group_metrics = {name: aggregate(items) for name, items in groups.items()}
    if "Long" in group_metrics and "Short" in group_metrics:
        long = group_metrics["Long"]
        short = group_metrics["Short"]
        group_metrics["PaperBalanced"] = {
            "num_expressions": int(long["num_expressions"]) + int(short["num_expressions"]),
            "J": (float(long["J"]) + float(short["J"])) / 2.0,
            "F": (float(long["F"]) + float(short["F"])) / 2.0,
            "J&F": (float(long["J&F"]) + float(short["J&F"])) / 2.0,
        }
    print_table(group_metrics)

    output = {
        "boundary_metric": "DAVIS-style seg2bmap and disk-dilation F, bound_th=0.008 by default",
        "meta": str(meta_path),
        "mask_dict": str(mask_dict_path),
        "pred_json": str(pred_json_path) if pred_json_path else None,
        "pred_root": str(pred_root) if pred_root else None,
        "boundary_th": args.boundary_th,
        "groups": group_metrics,
        "stats": dict(stats),
        "num_evaluated_expressions": len(per_sample),
        "num_expected_expressions": total_expressions,
        "per_sample": per_sample,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"out_json: {out_json}")
    print(f"stats: {dict(stats)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
