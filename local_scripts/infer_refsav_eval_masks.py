from __future__ import annotations

"""Run Sa2VA-Pro Ref-SAV inference and save per-frame prediction masks.

This mirrors the official Sa2VA RefVOS-style evaluation flow: read
meta_expressions_valid.json, run one prediction per expression, and write
mask PNGs under Annotations/<video_id>/<exp_id>/<frame_name>.png.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

try:
    import pycocotools.mask as mask_utils
except Exception as exc:  # pragma: no cover
    raise SystemExit("pycocotools is required. Install it in the active env first.") from exc

from sa2va_pro_factory import load_sa2va_pro_model


DEFAULT_DATA_ROOT = Path("/XYFS02/HDD_POOL/lzu_bmwang/lzu_bmwang_1/data/Sa2VA-data")
DEFAULT_EVAL_ROOT = DEFAULT_DATA_ROOT / "ref_sav_eval"
DEFAULT_META = DEFAULT_EVAL_ROOT / "meta_expressions_valid.json"
DEFAULT_VIDEO_ROOT = DEFAULT_EVAL_ROOT / "videos"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def official_prompt(caption: str) -> str:
    if "?" in caption:
        return f"<image>\n{caption} Please respond with a segmentation mask."
    return f"<image>\nPlease segment {caption}."


def expression_text(expr: dict[str, Any]) -> str:
    value = expr.get("exp", "")
    return value.strip() if isinstance(value, str) else ""


def load_video_frames(video_dir: Path, frame_names: list[str]) -> list[Image.Image]:
    frames: list[Image.Image] = []
    for frame_name in frame_names:
        path = video_dir / f"{frame_name}.jpg"
        if not path.is_file():
            raise FileNotFoundError(path)
        frames.append(Image.open(path).convert("RGB"))
    return frames


def normalize_prediction_masks(prediction_masks: Any) -> list[np.ndarray]:
    if not prediction_masks:
        return []

    first = prediction_masks[0] if isinstance(prediction_masks, (list, tuple)) else prediction_masks
    arr = np.asarray(first)
    if arr.ndim == 2:
        return [arr > 0]
    if arr.ndim == 3:
        return [item > 0 for item in arr]
    if arr.ndim == 4 and arr.shape[1] == 1:
        return [item[0] > 0 for item in arr]
    if arr.ndim == 4 and arr.shape[-1] == 1:
        return [item[..., 0] > 0 for item in arr]
    return []


def resize_bool_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask.astype(bool)
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    image = image.resize((shape[1], shape[0]), resample=Image.NEAREST)
    return np.asarray(image) > 0


def mask_to_png(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def mask_to_rle(mask: np.ndarray) -> dict[str, Any]:
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = encoded.get("counts")
    if isinstance(counts, bytes):
        encoded["counts"] = counts.decode("utf-8")
    return encoded


def existing_prediction_complete(ann_dir: Path, frame_names: list[str]) -> bool:
    return bool(frame_names) and all((ann_dir / f"{frame_name}.png").is_file() for frame_name in frame_names)


def load_existing_masks(ann_dir: Path, frame_names: list[str]) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for frame_name in frame_names:
        arr = np.asarray(Image.open(ann_dir / f"{frame_name}.png"))
        if arr.ndim == 3:
            arr = arr[..., 0]
        masks.append(arr > 0)
    return masks


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Ref-SAV mask inference.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--config-path", default=None)
    parser.add_argument("--sam2-ckpt", default=None)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--postfix-every", type=int, default=25)
    args = parser.parse_args()

    if args.sam2_ckpt:
        os.environ["SA2VA_PRO_SAM2_CKPT"] = args.sam2_ckpt

    meta_path = args.meta.resolve()
    video_root = args.video_root.resolve()
    out_root = args.out_root.resolve()
    ann_root = out_root / "Annotations"
    out_root.mkdir(parents=True, exist_ok=True)
    ann_root.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    print(f"Model: {args.model_path}")
    print(f"Config: {args.config_path}")
    print(f"SAM2 ckpt: {os.environ.get('SA2VA_PRO_SAM2_CKPT')}")
    print(f"Meta: {meta_path}")
    print(f"Video root: {video_root}")
    print(f"Output root: {out_root}")
    print(f"Device: {device}")

    model, processor = load_sa2va_pro_model(args.model_path, config_path=args.config_path)
    model.to(device)
    model.eval()

    meta = load_json(meta_path)
    videos = meta["videos"]
    samples: list[tuple[str, str, dict[str, Any], list[str]]] = []
    for video_id, video_item in videos.items():
        frame_names = sorted(str(frame) for frame in video_item.get("frames", []) if str(frame).strip())
        expressions = video_item.get("expressions", {})
        for exp_id in sorted(expressions.keys(), key=str):
            samples.append((video_id, str(exp_id), expressions[exp_id], frame_names))
            if args.limit is not None and len(samples) >= args.limit:
                break
        if args.limit is not None and len(samples) >= args.limit:
            break

    results: dict[str, dict[str, list[dict[str, Any]]]] = {}
    text_path = out_root / "prediction_texts.jsonl"
    stats = {
        "written": 0,
        "resumed": 0,
        "missing_caption": 0,
        "empty_prediction": 0,
        "short_prediction": 0,
        "long_prediction": 0,
        "resized_prediction": 0,
        "failed": 0,
    }

    print(f"Expressions to run: {len(samples)}")
    print("Progress: start")
    bar = tqdm(total=len(samples), unit="expr", desc="Ref-SAV-Infer", dynamic_ncols=True) if tqdm else None

    with text_path.open("w", encoding="utf-8") as text_file:
        for index, (video_id, exp_id, expr, frame_names) in enumerate(samples, start=1):
            if bar:
                bar.update(1)

            caption = expression_text(expr)
            if not caption:
                stats["missing_caption"] += 1
                continue

            video_dir = video_root / video_id
            ann_dir = ann_root / video_id / exp_id
            target_shape: tuple[int, int] | None = None

            try:
                if args.resume and existing_prediction_complete(ann_dir, frame_names):
                    masks = load_existing_masks(ann_dir, frame_names)
                    prediction_text = ""
                    stats["resumed"] += 1
                else:
                    frames = load_video_frames(video_dir, frame_names)
                    if frames:
                        width, height = frames[0].size
                        target_shape = (height, width)
                    prompt = official_prompt(caption)
                    with torch.inference_mode():
                        output = model.predict_forward(
                            video=frames,
                            text=prompt,
                            processor=processor,
                        )
                    prediction_text = str(output.get("prediction", ""))
                    masks = normalize_prediction_masks(output.get("prediction_masks", []))

                if target_shape is None:
                    first_frame = Image.open(video_dir / f"{frame_names[0]}.jpg")
                    width, height = first_frame.size
                    target_shape = (height, width)

                if not masks:
                    stats["empty_prediction"] += 1
                if len(masks) < len(frame_names):
                    stats["short_prediction"] += 1
                if len(masks) > len(frame_names):
                    stats["long_prediction"] += 1

                aligned_masks: list[np.ndarray] = []
                for local_i, frame_name in enumerate(frame_names):
                    if local_i < len(masks):
                        mask = masks[local_i]
                        if mask.shape != target_shape:
                            stats["resized_prediction"] += 1
                            mask = resize_bool_mask(mask, target_shape)
                    else:
                        mask = np.zeros(target_shape, dtype=bool)
                    mask_to_png(mask, ann_dir / f"{frame_name}.png")
                    aligned_masks.append(mask)

                results.setdefault(video_id, {})[exp_id] = [mask_to_rle(mask) for mask in aligned_masks]
                text_file.write(
                    json.dumps(
                        {
                            "video_id": video_id,
                            "exp_id": exp_id,
                            "prediction": prediction_text,
                            "num_pred_masks": len(masks),
                            "num_frames": len(frame_names),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                stats["written"] += 1

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as exc:
                stats["failed"] += 1
                text_file.write(
                    json.dumps(
                        {
                            "video_id": video_id,
                            "exp_id": exp_id,
                            "error": repr(exc),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            if bar and args.postfix_every > 0 and index % args.postfix_every == 0:
                bar.set_postfix(written=stats["written"], failed=stats["failed"])

    if bar:
        bar.close()

    results_path = out_root / "results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)

    summary = {
        "model_path": args.model_path,
        "config_path": args.config_path,
        "meta": str(meta_path),
        "video_root": str(video_root),
        "out_root": str(out_root),
        "results_json": str(results_path),
        "prediction_texts": str(text_path),
        "stats": stats,
        "num_expected": len(samples),
    }
    with (out_root / "inference_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"results_json: {results_path}")
    print(f"prediction_texts: {text_path}")
    print(f"summary_json: {out_root / 'inference_summary.json'}")
    print(f"stats: {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
