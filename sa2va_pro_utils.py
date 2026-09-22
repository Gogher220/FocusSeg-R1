from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


SEG_TOKEN = "[SEG]"
DEFAULT_VIDEO_FRAMES = 16
MASK_FIELD_CANDIDATES = (
    "segmentation_mask",
    "segmentation_masks",
    "segmentation_mask_path",
    "segmentation_mask_paths",
    "mask",
    "masks",
    "mask_path",
    "mask_paths",
    "target_mask",
    "target_masks",
)


def build_processor(
    base_model_name_or_path: str,
    seg_token: str = SEG_TOKEN,
    trust_remote_code: bool = True,
    use_fast: bool = False,
):
    processor = AutoProcessor.from_pretrained(
        base_model_name_or_path,
        trust_remote_code=trust_remote_code,
        use_fast=use_fast,
    )
    tokenizer = processor.tokenizer
    if tokenizer is None:
        raise ValueError("The base processor does not expose a tokenizer.")

    added = 0
    if tokenizer.convert_tokens_to_ids(seg_token) == tokenizer.unk_token_id:
        added = tokenizer.add_tokens([seg_token], special_tokens=False)

    tokenizer.padding_side = "left"
    processor.tokenizer = tokenizer
    return processor, added


def resolve_data_root(data_root: str | None = None) -> Path:
    if data_root:
        return Path(data_root).expanduser().resolve()
    env_root = os.environ.get("SA2VA_PRO_DATA_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path.cwd().resolve()


@lru_cache(maxsize=200000)
def _resolve_path_cached(path_like: str, data_root_str: str) -> str | None:
    root = Path(data_root_str)
    path = Path(path_like)

    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        rel = path_like[1:] if path_like.startswith("/") else path_like
        rel_path = Path(rel)
        candidates.extend(
            [
                root / rel_path,
                root / "coco" / rel_path,
                root / "coco" / "train2017" / rel_path,
                root / "coco" / "val2017" / rel_path,
                root / "coco_2017" / rel_path,
                root / "coco_2017" / "train2017" / rel_path,
                root / "coco_2017" / "val2017" / rel_path,
                root / "coco_2014" / rel_path,
                root / "coco_2014" / "train2014" / rel_path,
                root / "coco_2014" / "val2014" / rel_path,
                root / "llava_data" / rel_path,
                root / "glamm_data" / rel_path,
                root / "glamm_data" / "GranDf_HA_images" / "train" / rel_path,
                root / "glamm_data" / "GranDf_HA_images" / "val_test" / rel_path,
                root / "GranDf_HA_images" / "train" / rel_path,
                root / "GranDf_HA_images" / "val_test" / rel_path,
                root / "osprey-724k" / rel_path,
                root / "ref_seg" / rel_path,
                root / "video_datas" / rel_path,
                root / "flikcr_30k" / "train" / rel_path,
            ]
        )

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return str(candidate.resolve())
    return None


def resolve_path(path_like: str | None, data_root: str | Path | None = None) -> Path | None:
    if not path_like:
        return None
    root = resolve_data_root(str(data_root) if data_root else None)
    resolved = _resolve_path_cached(str(path_like), str(root))
    if resolved is None:
        return None
    return Path(resolved)


def load_image(path_like: str | Path) -> Image.Image:
    return Image.open(path_like).convert("RGB")


def _sample_video_frames_from_directory(frame_dir: str | Path, nframes: int) -> list[Image.Image]:
    frame_dir = Path(frame_dir)
    if not frame_dir.is_dir():
        return []

    frame_paths = [
        path for path in sorted(frame_dir.iterdir())
        if path.is_file() and path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    ]
    if not frame_paths:
        return []

    frame_ids = np.linspace(0, len(frame_paths) - 1, min(nframes, len(frame_paths)), dtype=int)
    frames: list[Image.Image] = []
    for frame_id in frame_ids:
        idx = min(int(frame_id), len(frame_paths) - 1)
        frames.append(load_image(frame_paths[idx]))
    return frames


def _load_video_frames_by_names(frame_dir: str | Path, frame_names: Iterable[Any]) -> list[Image.Image]:
    frame_dir = Path(frame_dir)
    if not frame_dir.is_dir():
        return []

    frames: list[Image.Image] = []
    image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    for frame_name in frame_names:
        name = str(frame_name).strip()
        if not name:
            continue
        path = frame_dir / name
        candidates = [path] if path.suffix else [frame_dir / f"{name}{ext}" for ext in image_exts]
        for candidate in candidates:
            if candidate.is_file():
                frames.append(load_image(candidate))
                break
    return frames


def _sample_video_frames_with_cv2(video_path: str | Path, nframes: int) -> list[Image.Image]:
    if cv2 is None:
        return []

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    frame_ids = np.linspace(0, total_frames - 1, min(nframes, total_frames), dtype=int)
    frames: list[Image.Image] = []
    for frame_id in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))
    cap.release()
    return frames


def _sample_video_frames_with_av(video_path: str | Path, nframes: int) -> list[Image.Image]:
    try:
        import av  # type: ignore
    except Exception:
        return []

    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        total_frames = stream.frames
        decoded_frames = []
        for frame in container.decode(video=0):
            decoded_frames.append(frame)

        if not decoded_frames:
            return []

        total_frames = len(decoded_frames) if total_frames <= 0 else total_frames
        frame_ids = np.linspace(0, total_frames - 1, min(nframes, total_frames), dtype=int)
        frames: list[Image.Image] = []
        for frame_id in frame_ids:
            idx = min(int(frame_id), len(decoded_frames) - 1)
            rgb = decoded_frames[idx].to_rgb().to_ndarray()
            frames.append(Image.fromarray(rgb))
        return frames
    finally:
        container.close()


def _sample_video_frames_with_imageio(video_path: str | Path, nframes: int) -> list[Image.Image]:
    try:
        import imageio.v3 as iio  # type: ignore
    except Exception:
        return []

    try:
        frames_np = iio.imread(str(video_path), index=None)
    except Exception:
        return []

    if frames_np is None or len(frames_np) == 0:
        return []

    total_frames = len(frames_np)
    frame_ids = np.linspace(0, total_frames - 1, min(nframes, total_frames), dtype=int)
    frames: list[Image.Image] = []
    for frame_id in frame_ids:
        idx = min(int(frame_id), total_frames - 1)
        frames.append(Image.fromarray(frames_np[idx]).convert("RGB"))
    return frames


def sample_video_frames(video_path: str | Path, nframes: int = DEFAULT_VIDEO_FRAMES) -> list[Image.Image]:
    video_path = Path(video_path)
    if video_path.is_dir():
        frames = _sample_video_frames_from_directory(video_path, nframes)
        if frames:
            return frames

    for reader in (
        _sample_video_frames_with_cv2,
        _sample_video_frames_with_av,
        _sample_video_frames_with_imageio,
    ):
        frames = reader(video_path, nframes)
        if frames:
            return frames
    raise ValueError(
        f"Failed to decode video frames from {video_path}. "
        "Install opencv-python-headless, pyav, or imageio with ffmpeg support."
    )


def load_visual_inputs(example: dict[str, Any], data_root: str | Path | None = None) -> tuple[str, Image.Image | list[Image.Image]]:
    data_type = example.get("data_type", "image")
    path_value = example.get("path") or example.get("image_path") or example.get("video_path")
    resolved = resolve_path(path_value, data_root)
    if resolved is None:
        raise ValueError("The dataset example does not contain a visual path.")

    if data_type == "video":
        nframes = int(example.get("nframes", DEFAULT_VIDEO_FRAMES))
        frame_names = example.get("frame_names") or []
        if resolved.is_dir() and frame_names:
            named_frames = _load_video_frames_by_names(resolved, list(frame_names)[:nframes])
            if named_frames:
                return data_type, named_frames
        return data_type, sample_video_frames(resolved, nframes=nframes)
    return "image", load_image(resolved)


def _load_mask_file(path_like: str | Path) -> np.ndarray:
    path = Path(path_like)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        mask = np.load(path)
    elif suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
        mask = np.array(Image.open(path))
    else:
        raise ValueError(f"Unsupported mask format: {path}")
    return np.asarray(mask)


def _normalize_mask_array(mask: np.ndarray) -> torch.Tensor:
    mask = np.asarray(mask)
    if mask.ndim == 2:
        mask = mask[None, ...]
    elif mask.ndim == 3 and mask.shape[-1] in {3, 4}:
        mask = mask[..., 0]
        mask = mask[None, ...]
    elif mask.ndim == 3 and mask.shape[0] != 1:
        pass
    elif mask.ndim == 4 and mask.shape[-1] in {3, 4}:
        mask = mask[..., 0]
    tensor = torch.as_tensor(mask, dtype=torch.float32)
    if tensor.max() > 1:
        tensor = (tensor > 0).float()
    return tensor


def _load_mask_item(item: Any, data_root: str | Path | None = None) -> torch.Tensor | None:
    if isinstance(item, (str, Path)):
        resolved = resolve_path(str(item), data_root)
        if resolved is None or not resolved.exists():
            return None
        return _normalize_mask_array(_load_mask_file(resolved))

    if isinstance(item, dict):
        path_value = item.get('path') or item.get('mask_path')
        if not path_value:
            return None
        resolved = resolve_path(str(path_value), data_root)
        if resolved is None or not resolved.exists():
            return None
        mask = np.asarray(_load_mask_file(resolved))
        obj_id = item.get('obj_id', item.get('instance_id', item.get('label_id')))
        if obj_id is not None:
            try:
                obj_value = int(obj_id)
                mask = (mask == obj_value).astype(np.float32)
            except Exception:
                pass
        return _normalize_mask_array(mask)

    return _normalize_mask_array(np.asarray(item))


def _iter_mask_values(value: Any) -> Iterable[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return value
    return [value]


def load_segmentation_targets(
    example: dict[str, Any],
    data_root: str | Path | None = None,
) -> torch.Tensor | None:
    for field in MASK_FIELD_CANDIDATES:
        if field not in example:
            continue
        value = example[field]
        tensors = []
        for item in _iter_mask_values(value):
            tensor = _load_mask_item(item, data_root)
            if tensor is None:
                continue
            tensors.append(tensor)
        if not tensors:
            continue
        if len(tensors) == 1:
            return tensors[0]
        return torch.cat(tensors, dim=0)
    return None


def resize_gt_masks(gt_masks: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    if gt_masks.ndim == 2:
        gt_masks = gt_masks.unsqueeze(0)
    if gt_masks.ndim != 3:
        raise ValueError(f"Expected gt masks with 3 dims after normalization, got {gt_masks.shape}")
    gt_masks = gt_masks.unsqueeze(1)
    resized = F.interpolate(gt_masks, size=target_hw, mode="nearest")
    return resized[:, 0]


def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = logits.sigmoid()
    probs = probs.flatten(1)
    targets = targets.flatten(1)
    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def bce_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, targets)


def mask_iou_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    preds = (logits.sigmoid() > 0.5).float()
    targets = (targets > 0.5).float()
    preds = preds.flatten(1)
    targets = targets.flatten(1)
    intersection = (preds * targets).sum(dim=1)
    union = ((preds + targets) > 0).float().sum(dim=1)
    return ((intersection + eps) / (union + eps)).mean()


def shuffle_video_frames(frames: list[Image.Image]) -> list[Image.Image]:
    if len(frames) <= 1:
        return list(frames)
    shuffled = list(frames)
    np.random.shuffle(shuffled)
    return shuffled


def _sample_random_square_box(
    frame_size: tuple[int, int],
    occ_ratio_min: float,
    occ_ratio_max: float,
) -> tuple[int, int, int, int] | None:
    width, height = frame_size
    if width <= 0 or height <= 0:
        return None

    min_ratio = max(0.0, min(float(occ_ratio_min), 1.0))
    max_ratio = max(min_ratio, min(float(occ_ratio_max), 1.0))
    if max_ratio <= 0.0:
        return None

    occ_ratio = float(np.random.uniform(min_ratio, max_ratio))
    side = int(round(np.sqrt(occ_ratio * width * height)))
    side = max(1, min(side, width, height))

    x0 = 0 if side >= width else int(np.random.randint(0, width - side + 1))
    y0 = 0 if side >= height else int(np.random.randint(0, height - side + 1))
    return x0, y0, x0 + side, y0 + side


def _select_random_frame_indices(num_frames: int, occ_frame_ratio: float) -> list[int]:
    if num_frames <= 0:
        return []
    ratio = max(0.0, min(float(occ_frame_ratio), 1.0))
    if ratio <= 0.0:
        return []
    num_selected = max(1, int(round(num_frames * ratio)))
    num_selected = min(num_selected, num_frames)
    return sorted(np.random.choice(list(range(num_frames)), size=num_selected, replace=False).tolist())


def apply_random_spatial_occlusion_to_video(
    frames: list[Image.Image],
    occ_ratio_min: float = 0.10,
    occ_ratio_max: float = 0.30,
    occ_frame_ratio: float = 0.4,
) -> tuple[list[Image.Image], dict[str, Any]]:
    metadata: dict[str, Any] = {
        "applied": False,
        "selected_frame_indices": [],
        "occlusion_boxes": [],
        "mode": "random_square",
    }
    if not frames:
        return list(frames), metadata

    selected_indices = _select_random_frame_indices(len(frames), occ_frame_ratio)
    if not selected_indices:
        return list(frames), metadata

    masked_frames = [frame.copy() for frame in frames]
    occlusion_boxes: list[tuple[int, int, int, int]] = []
    applied_indices: list[int] = []
    for frame_index in selected_indices:
        occ_box = _sample_random_square_box(
            masked_frames[frame_index].size,
            occ_ratio_min=occ_ratio_min,
            occ_ratio_max=occ_ratio_max,
        )
        if occ_box is None:
            continue
        frame_np = np.asarray(masked_frames[frame_index]).copy()
        ox0, oy0, ox1, oy1 = occ_box
        frame_np[oy0:oy1, ox0:ox1, :] = 0
        masked_frames[frame_index] = Image.fromarray(frame_np)
        occlusion_boxes.append(occ_box)
        applied_indices.append(frame_index)

    if not occlusion_boxes:
        return list(frames), metadata

    metadata["applied"] = True
    metadata["selected_frame_indices"] = applied_indices
    metadata["occlusion_boxes"] = occlusion_boxes
    return masked_frames, metadata


def apply_random_spatial_occlusion_to_image(
    image: Image.Image,
    occ_ratio_min: float = 0.10,
    occ_ratio_max: float = 0.30,
) -> tuple[Image.Image, dict[str, Any]]:
    frames, metadata = apply_random_spatial_occlusion_to_video(
        [image],
        occ_ratio_min=occ_ratio_min,
        occ_ratio_max=occ_ratio_max,
        occ_frame_ratio=1.0,
    )
    return frames[0], metadata


def _resize_mask_to_frame(mask: np.ndarray, frame_size: tuple[int, int]) -> np.ndarray:
    width, height = frame_size
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    resized = Image.fromarray(mask_uint8).resize((width, height), Image.NEAREST)
    return np.asarray(resized) > 0


def _mask_index_for_frame(frame_index: int, num_frames: int, num_masks: int) -> int:
    if num_masks <= 1 or num_frames <= 1:
        return 0
    ratio = frame_index / max(num_frames - 1, 1)
    return min(int(round(ratio * (num_masks - 1))), num_masks - 1)


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _sample_occlusion_box(
    bbox: tuple[int, int, int, int],
    keep_ratio: float,
    occ_ratio_min: float,
    occ_ratio_max: float,
) -> tuple[int, int, int, int] | None:
    x0, y0, x1, y1 = bbox
    box_w = max(1, x1 - x0)
    box_h = max(1, y1 - y0)
    box_area = box_w * box_h
    max_occ_ratio = max(0.0, min(float(occ_ratio_max), 1.0 - float(keep_ratio)))
    min_occ_ratio = max(0.0, min(float(occ_ratio_min), max_occ_ratio))
    if max_occ_ratio <= 0.0:
        return None

    occ_ratio = float(np.random.uniform(min_occ_ratio, max_occ_ratio))
    occ_area = max(1.0, occ_ratio * box_area)
    aspect = float(np.random.uniform(0.75, 1.33))
    occ_w = int(np.sqrt(occ_area * aspect))
    occ_h = int(np.sqrt(occ_area / max(aspect, 1e-6)))
    occ_w = max(1, min(occ_w, box_w))
    occ_h = max(1, min(occ_h, box_h))

    if occ_w >= box_w:
        ox0 = x0
    else:
        ox0 = int(np.random.randint(x0, x1 - occ_w + 1))
    if occ_h >= box_h:
        oy0 = y0
    else:
        oy0 = int(np.random.randint(y0, y1 - occ_h + 1))
    return ox0, oy0, ox0 + occ_w, oy0 + occ_h


def apply_spatial_occlusion_to_video(
    frames: list[Image.Image],
    target_masks: torch.Tensor | np.ndarray | None,
    keep_ratio: float = 0.6,
    occ_ratio_min: float = 0.15,
    occ_ratio_max: float = 0.35,
    occ_frame_ratio: float = 0.5,
) -> tuple[list[Image.Image], dict[str, Any]]:
    metadata: dict[str, Any] = {
        "applied": False,
        "selected_frame_indices": [],
        "occlusion_boxes": [],
    }
    if not frames or target_masks is None:
        return list(frames), metadata

    masks = target_masks.detach().cpu().numpy() if isinstance(target_masks, torch.Tensor) else np.asarray(target_masks)
    if masks.ndim == 2:
        masks = masks[None, ...]
    elif masks.ndim != 3:
        return list(frames), metadata

    candidate_indices: list[int] = []
    resized_masks: dict[int, np.ndarray] = {}
    for frame_index, frame in enumerate(frames):
        mask_index = _mask_index_for_frame(frame_index, len(frames), masks.shape[0])
        resized_mask = _resize_mask_to_frame(masks[mask_index], frame.size)
        if resized_mask.any():
            candidate_indices.append(frame_index)
            resized_masks[frame_index] = resized_mask

    if not candidate_indices:
        return list(frames), metadata

    num_selected = max(1, int(round(len(candidate_indices) * float(occ_frame_ratio))))
    num_selected = min(num_selected, len(candidate_indices))
    selected_indices = sorted(np.random.choice(candidate_indices, size=num_selected, replace=False).tolist())

    masked_frames = [frame.copy() for frame in frames]
    occlusion_boxes: list[tuple[int, int, int, int]] = []
    for frame_index in selected_indices:
        bbox = _bbox_from_mask(resized_masks[frame_index])
        if bbox is None:
            continue
        occ_box = _sample_occlusion_box(
            bbox=bbox,
            keep_ratio=keep_ratio,
            occ_ratio_min=occ_ratio_min,
            occ_ratio_max=occ_ratio_max,
        )
        if occ_box is None:
            continue

        frame_np = np.asarray(masked_frames[frame_index]).copy()
        ox0, oy0, ox1, oy1 = occ_box
        frame_np[oy0:oy1, ox0:ox1, :] = 0
        masked_frames[frame_index] = Image.fromarray(frame_np)
        occlusion_boxes.append(occ_box)

    if not occlusion_boxes:
        return list(frames), metadata

    metadata["applied"] = True
    metadata["selected_frame_indices"] = selected_indices
    metadata["occlusion_boxes"] = occlusion_boxes
    return masked_frames, metadata


def apply_spatial_occlusion_to_image(
    image: Image.Image,
    target_masks: torch.Tensor | np.ndarray | None,
    keep_ratio: float = 0.6,
    occ_ratio_min: float = 0.15,
    occ_ratio_max: float = 0.35,
) -> tuple[Image.Image, dict[str, Any]]:
    frames, metadata = apply_spatial_occlusion_to_video(
        [image],
        target_masks,
        keep_ratio=keep_ratio,
        occ_ratio_min=occ_ratio_min,
        occ_ratio_max=occ_ratio_max,
        occ_frame_ratio=1.0,
    )
    return frames[0], metadata
