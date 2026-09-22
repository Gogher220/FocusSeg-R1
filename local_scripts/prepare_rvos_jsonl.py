from __future__ import annotations

import argparse
import json
from pathlib import Path


IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def normalize_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace('\\', '/')
    except ValueError:
        return str(path).replace('\\', '/')


def first_existing_dir(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def first_existing_file(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def resolve_frame_dir(dataset_root: Path, dataset_name: str, split: str, video_id: str) -> Path | None:
    candidates = [
        dataset_root / 'video_datas' / dataset_name / split / 'JPEGImages' / video_id,
        dataset_root / 'video_datas' / dataset_name / split / 'JPEGImages' / '480p' / video_id,
        dataset_root / 'video_datas' / dataset_name / 'train' / 'JPEGImages' / video_id,
        dataset_root / 'video_datas' / dataset_name / 'train' / 'JPEGImages' / '480p' / video_id,
        dataset_root / 'video_datas' / dataset_name / 'JPEGImages' / video_id,
        dataset_root / 'video_datas' / dataset_name / 'JPEGImages' / '480p' / video_id,
        dataset_root / 'video_datas' / dataset_name / split / 'images' / video_id,
        dataset_root / 'video_datas' / dataset_name / 'images' / video_id,
    ]
    return first_existing_dir(candidates)


def resolve_mask_dir(dataset_root: Path, dataset_name: str, split: str, video_id: str) -> Path | None:
    candidates = [
        dataset_root / 'video_datas' / dataset_name / split / 'Annotations' / video_id,
        dataset_root / 'video_datas' / dataset_name / split / 'Annotations' / '480p' / video_id,
        dataset_root / 'video_datas' / dataset_name / 'train' / 'Annotations' / video_id,
        dataset_root / 'video_datas' / dataset_name / 'train' / 'Annotations' / '480p' / video_id,
        dataset_root / 'video_datas' / dataset_name / 'Annotations' / video_id,
        dataset_root / 'video_datas' / dataset_name / 'Annotations' / '480p' / video_id,
        dataset_root / 'video_datas' / dataset_name / split / 'masks' / video_id,
        dataset_root / 'video_datas' / dataset_name / 'masks' / video_id,
    ]
    return first_existing_dir(candidates)


def resolve_mask_path(mask_dir: Path, frame_name: str) -> Path | None:
    candidates = [mask_dir / f'{frame_name}.png', mask_dir / f'{frame_name}.jpg', mask_dir / f'{frame_name}.jpeg', mask_dir / f'{frame_name}.npy']
    return first_existing_file(candidates)


def build_example(
    *,
    data_root: Path,
    dataset_name: str,
    split: str,
    video_id: str,
    expression: str,
    obj_id: str,
    frame_dir: Path,
    mask_dir: Path,
    frame_names: list[str],
    nframes: int,
) -> dict | None:
    if not frame_names:
        return None

    sampled = frame_names[: min(nframes, len(frame_names))]
    mask_specs = []
    for frame_name in sampled:
        mask_path = resolve_mask_path(mask_dir, frame_name)
        if mask_path is None:
            continue
        mask_specs.append({'path': normalize_path(mask_path, data_root), 'obj_id': int(obj_id)})
    if not mask_specs:
        return None

    return {
        'problem': f'Segment the object referred by: {expression}',
        'problem_type': 'segmentation',
        'data_type': 'video',
        'path': normalize_path(frame_dir, data_root),
        'nframes': len(sampled),
        'frame_names': sampled,
        'mask_paths': mask_specs,
        'process': '<think>I locate the referred target across the sampled video frames and prepare the segmentation output.</think>',
        'solution': '<answer>target object [SEG]</answer>',
        'source_dataset': dataset_name,
        'source_split': split,
        'video_id': video_id,
        'obj_id': int(obj_id),
    }


def convert_meta_file(meta_path: Path, data_root: Path, out_path: Path, nframes: int) -> int:
    dataset_name = meta_path.parents[2].name
    split = meta_path.parent.name
    with meta_path.open('r', encoding='utf-8') as f:
        payload = json.load(f)

    videos = payload.get('videos', {})
    count = 0
    with out_path.open('w', encoding='utf-8') as out_file:
        for video_id, video_info in videos.items():
            frame_dir = resolve_frame_dir(data_root, dataset_name, split, video_id)
            mask_dir = resolve_mask_dir(data_root, dataset_name, split, video_id)
            if frame_dir is None or mask_dir is None:
                continue

            frame_names = [str(x) for x in video_info.get('frames', []) if str(x).strip()]
            expressions = video_info.get('expressions', {})
            for expr_info in expressions.values():
                expression = str(expr_info.get('exp', '')).strip()
                obj_id = str(expr_info.get('obj_id', '')).strip()
                if not expression or not obj_id:
                    continue
                example = build_example(
                    data_root=data_root,
                    dataset_name=dataset_name,
                    split=split,
                    video_id=video_id,
                    expression=expression,
                    obj_id=obj_id,
                    frame_dir=frame_dir,
                    mask_dir=mask_dir,
                    frame_names=frame_names,
                    nframes=nframes,
                )
                if example is None:
                    continue
                out_file.write(json.dumps(example, ensure_ascii=False) + '\n')
                count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description='Convert Sa2VA-Training RVOS/DAVIS metadata into Sa2VA_Pro JSONL.')
    parser.add_argument('--train_root', type=str, required=True, help='Root path of Sa2VA-Training.')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory for converted JSONL files.')
    parser.add_argument('--nframes', type=int, default=16, help='Maximum number of frames kept per sample.')
    args = parser.parse_args()

    train_root = Path(args.train_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_files = sorted(train_root.rglob('meta_expressions.json'))
    merged_path = out_dir / 'train_sft_all.jsonl'
    stats_path = out_dir / 'prepare_stats.json'

    merged_count = 0
    source_counts: dict[str, int] = {}
    with merged_path.open('w', encoding='utf-8') as merged_file:
        for meta_file in meta_files:
            rel_name = str(meta_file.relative_to(train_root)).replace('\\', '/')
            per_source_path = out_dir / (meta_file.parents[2].name + '_' + meta_file.parent.name + '.jsonl')
            count = convert_meta_file(meta_file, train_root, per_source_path, args.nframes)
            if count == 0:
                per_source_path.unlink(missing_ok=True)
                continue
            source_counts[rel_name] = count
            merged_count += count
            with per_source_path.open('r', encoding='utf-8') as src:
                for line in src:
                    merged_file.write(line)

    stats = {
        'train_root': str(train_root),
        'out_dir': str(out_dir),
        'nframes': int(args.nframes),
        'total_examples': merged_count,
        'sources': source_counts,
    }
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
