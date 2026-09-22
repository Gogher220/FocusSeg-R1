from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
from PIL import Image

from sa2va_pro_factory import load_sa2va_pro_model


TASKS = [
    "existence",
    "count",
    "position",
    "color",
    "posters",
    "celebrity",
    "scene",
    "landmark",
    "artwork",
    "OCR",
    "commonsense_reasoning",
    "numerical_calculation",
    "text_translation",
    "code_reasoning",
]


def extract_answer(text: str) -> str:
    text = text.strip()
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.S)
    if match:
        return match.group(1).strip()
    text = text.replace("<|im_end|>", "").strip()
    text = text.split("\n")[0].strip()
    return text


def parse_question_file(txt_path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = re.split(r"\t+", line)
            if len(parts) >= 2:
                question = parts[0].strip()
                answer = parts[-1].strip()
                rows.append((question, answer))
                continue
            if "Please answer yes or no." in line:
                idx = line.rfind("Please answer yes or no.")
                question = line[: idx + len("Please answer yes or no.")].strip()
                answer = line[idx + len("Please answer yes or no.") :].strip()
                rows.append((question, answer))
    return rows


def find_image_for_txt(txt_path: Path) -> Path:
    stem = txt_path.stem
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        image_path = txt_path.with_suffix(ext)
        if image_path.exists():
            return image_path
    raise FileNotFoundError(f"image not found for {txt_path}")


def iter_task_files(task_dir: Path) -> list[Path]:
    return sorted(task_dir.rglob("*.txt"))


def run_one_task(task: str, task_dir: Path, out_path: Path, model, processor, limit: int = 0):
    txt_files = iter_task_files(task_dir)
    if limit > 0:
        txt_files = txt_files[:limit]

    with out_path.open("w", encoding="utf-8") as fout:
        for txt_path in txt_files:
            try:
                image_path = find_image_for_txt(txt_path)
            except FileNotFoundError:
                continue

            qa_pairs = parse_question_file(txt_path)
            if not qa_pairs:
                continue

            image = Image.open(image_path).convert("RGB")
            for question, gt in qa_pairs:
                with torch.inference_mode():
                    ret = model.predict_forward(
                        image=image,
                        text=question,
                        processor=processor,
                    )
                pred = extract_answer(ret["prediction"])
                fout.write(f"{image_path.name}\t{question}\t{gt}\t{pred}\n")

    print(f"saved: {out_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Run MME evaluation for Sa2VA_Pro checkpoints.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--mme_root", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    model_path = Path(args.model_path).expanduser().resolve()
    mme_root = Path(args.mme_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model, processor = load_sa2va_pro_model(str(model_path))
    model = model.eval().cuda()
    if hasattr(model, "torch_dtype"):
        model = model.to(dtype=torch.bfloat16)

    tasks = [args.task] if args.task else TASKS
    for task in tasks:
        task_dir = mme_root / task
        if not task_dir.exists():
            continue
        out_path = out_dir / f"{task}.txt"
        run_one_task(task, task_dir, out_path, model, processor, limit=args.limit)


if __name__ == "__main__":
    main()
