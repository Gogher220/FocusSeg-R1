from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MASK_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".npy"}


def iter_json_files(root: Path):
    for path in root.rglob("*.json"):
        yield path
    for path in root.rglob("*.jsonl"):
        yield path


def read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    records.append(obj)
        return records

    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        for key in ["data", "annotations", "images", "items", "samples"]:
            value = obj.get(key)
            if isinstance(value, list) and value and isinstance(value[0], dict):
                return value
        return [obj]
    return []


def should_skip_in_debug(path: Path, max_per_file: int) -> bool:
    if max_per_file <= 0:
        return False
    path_str = str(path).replace("\\", "/").lower()
    # Large raw annotation dumps are slow to json.loads in debug mode.
    slow_keywords = [
        "llava_images_full",
        "llava_data_pretrain",
        "osprey-724k",
        "glamm_data",
    ]
    return any(keyword in path_str for keyword in slow_keywords)


def looks_like_image(value: str) -> bool:
    return Path(value).suffix.lower() in IMAGE_EXTS


def looks_like_video(value: str) -> bool:
    return Path(value).suffix.lower() in VIDEO_EXTS


def looks_like_mask(value: str) -> bool:
    return Path(value).suffix.lower() in MASK_EXTS


def find_first_string(record: dict[str, Any], candidates: list[str]) -> str | None:
    for key in candidates:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def find_first_list(record: dict[str, Any], candidates: list[str]) -> list[str] | None:
    for key in candidates:
        value = record.get(key)
        if isinstance(value, list):
            out = [x.strip() for x in value if isinstance(x, str) and x.strip()]
            if out:
                return out
    return None


def infer_visual_path(record: dict[str, Any]) -> tuple[str | None, str | None]:
    string_values = [v.strip() for v in record.values() if isinstance(v, str) and v.strip()]

    path_value = find_first_string(
        record,
        [
            "path",
            "image",
            "image_path",
            "img_path",
            "file_name",
            "filename",
            "video",
            "video_path",
        ],
    )

    if path_value:
        if looks_like_video(path_value):
            return "video", path_value
        if looks_like_image(path_value):
            return "image", path_value

    for value in string_values:
        if looks_like_video(value):
            return "video", value
    for value in string_values:
        if looks_like_image(value):
            return "image", value
    return None, None


def infer_mask_paths(record: dict[str, Any]) -> tuple[str | None, list[str] | None]:
    mask_list = find_first_list(
        record,
        [
            "mask_paths",
            "masks",
            "segmentation_mask_paths",
            "target_masks",
        ],
    )
    if mask_list:
        mask_list = [x for x in mask_list if looks_like_mask(x)]
        if mask_list:
            return None, mask_list

    mask_path = find_first_string(
        record,
        [
            "mask_path",
            "mask",
            "segmentation_mask",
            "segmentation_mask_path",
            "target_mask",
        ],
    )
    if mask_path and looks_like_mask(mask_path):
        return mask_path, None
    return None, None


def infer_problem(record: dict[str, Any]) -> str | None:
    question = find_first_string(
        record,
        [
            "problem",
            "question",
            "query",
            "instruction",
            "text",
            "sent",
            "expression",
            "refer_sentence",
            "caption",
        ],
    )
    if question:
        return question
    return None


def infer_solution(record: dict[str, Any], has_mask: bool) -> str | None:
    answer = find_first_string(
        record,
        [
            "solution",
            "answer",
            "response",
            "output",
            "label",
            "caption",
        ],
    )
    if answer:
        if "<answer>" not in answer:
            answer = f"<answer>{answer}</answer>"
        if has_mask and "[SEG]" not in answer:
            answer = answer.replace("</answer>", " [SEG]</answer>")
        return answer

    if has_mask:
        return "<answer>target object [SEG]</answer>"
    return None


def infer_process(record: dict[str, Any], has_mask: bool) -> str:
    process = find_first_string(record, ["process", "reasoning", "rationale", "cot", "explanation"])
    if process:
        if "<think>" not in process:
            process = f"<think>{process}</think>"
        return process

    if has_mask:
        return "<think>I locate the referred target and prepare the segmentation output.</think>"
    return "<think>I inspect the visual content and derive the final answer.</think>"


def infer_problem_type(answer: str | None, has_mask: bool = False) -> str:
    if has_mask:
        return "segmentation"
    if not answer:
        return "free-form"
    lower = answer.lower()
    if any(tag in lower for tag in ["yes", "no"]):
        return "free-form"
    return "free-form"


def normalize_path(path_value: str) -> str:
    return path_value.replace("\\", "/")


def convert_record(record: dict[str, Any]) -> dict[str, Any] | None:
    data_type, path_value = infer_visual_path(record)
    if not data_type or not path_value:
        return None

    mask_path, mask_paths = infer_mask_paths(record)
    has_mask = bool(mask_path or mask_paths)
    problem = infer_problem(record)
    if not problem:
        return None

    solution = infer_solution(record, has_mask=has_mask)
    if not solution:
        return None

    output = {
        "problem": problem,
        "problem_type": infer_problem_type(solution, has_mask=has_mask),
        "data_type": data_type,
        "path": normalize_path(path_value),
        "process": infer_process(record, has_mask=has_mask),
        "solution": solution,
    }

    if data_type == "video":
        output["nframes"] = int(record.get("nframes", record.get("num_frames", 16)))

    if mask_path:
        output["mask_path"] = normalize_path(mask_path)
    if mask_paths:
        output["mask_paths"] = [normalize_path(x) for x in mask_paths]

    return output


def main():
    parser = argparse.ArgumentParser(description="Convert Sa2VA-Training style annotations into Sa2VA_Pro JSONL.")
    parser.add_argument("--train_root", type=str, required=True, help="Root path of Sa2VA-Training.")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for converted JSONL files.")
    parser.add_argument("--max_per_file", type=int, default=0, help="Optional per-source cap for debugging.")
    parser.add_argument(
        "--data_types",
        type=str,
        default="image,video",
        help="Comma separated data types to keep. Example: image or image,video",
    )
    args = parser.parse_args()

    train_root = Path(args.train_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    keep_types = {x.strip() for x in args.data_types.split(",") if x.strip()}

    merged_path = out_dir / "train_sft_all.jsonl"
    stats_path = out_dir / "prepare_stats.json"

    source_counts: dict[str, int] = {}
    total = 0

    with merged_path.open("w", encoding="utf-8") as merged_file:
        for ann_path in sorted(iter_json_files(train_root)):
            rel_source = str(ann_path.relative_to(train_root)).replace("\\", "/")
            if should_skip_in_debug(ann_path, args.max_per_file):
                continue
            records = read_records(ann_path)
            if not records:
                continue

            converted_count = 0
            per_source_path = out_dir / (ann_path.stem + ".jsonl")
            with per_source_path.open("w", encoding="utf-8") as per_source_file:
                for record in records:
                    example = convert_record(record)
                    if example is None:
                        continue
                    if example["data_type"] not in keep_types:
                        continue
                    line = json.dumps(example, ensure_ascii=False)
                    per_source_file.write(line + "\n")
                    merged_file.write(line + "\n")
                    converted_count += 1
                    total += 1
                    if args.max_per_file > 0 and converted_count >= args.max_per_file:
                        break

            if converted_count == 0:
                per_source_path.unlink(missing_ok=True)
                continue
            source_counts[rel_source] = converted_count

    stats = {
        "train_root": str(train_root),
        "out_dir": str(out_dir),
        "total_examples": total,
        "sources": source_counts,
    }
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
