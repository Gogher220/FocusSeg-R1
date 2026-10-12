"""Strict, expression-level RefVOS samples aligned with Sa2VA's mask mapping.

The metadata index is inexpensive to load. Large mask dictionaries are loaded only
when a caller explicitly requests a sample on a compute node; never during import.
"""

from __future__ import annotations

import json
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class RefVOSExpression:
    source: str
    video_id: str
    expression_id: str
    text: str
    frame_names: tuple[str, ...]
    mask_anno_ids: tuple[str, ...]
    object_ids: tuple[str, ...]


@dataclass
class RefVOSSample:
    expression: RefVOSExpression
    frame_indices: tuple[int, ...]
    images: list[Any]  # PIL.Image.Image; imported lazily on the compute node.
    masks: Any  # numpy.ndarray, [frames, height, width], bool.


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    return (str(value),)


def read_refvos_index(
    meta_path: str | Path, *, source: str, synthetic_anno_ids: bool = False,
    skip_missing_mask_annotation_id: bool = False,
    excluded_missing_anno_ids: list[str] | None = None,
) -> list[RefVOSExpression]:
    """Read training metadata; synthetic IDs follow the official traversal order.

    Ref-YTVOS has no annotation ID in its expression metadata. Sa2VA enumerates
    video entries in JSON insertion order and expression IDs in sorted order.
    Its mask dictionary is indexed by that running expression count.
    """
    with Path(meta_path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    videos = payload.get("videos")
    if not isinstance(videos, dict) or not videos:
        raise ValueError(f"Expected a nonempty 'videos' mapping: {meta_path}")

    expressions: list[RefVOSExpression] = []
    running_anno_id = 0
    for video_id, video in videos.items():
        frames = tuple(sorted(str(name) for name in video.get("frames", [])))
        if not frames:
            raise ValueError(f"No frames for {source}/{video_id}")
        raw_expressions = video.get("expressions")
        if not isinstance(raw_expressions, dict):
            raise ValueError(f"No expression mapping for {source}/{video_id}")
        for expression_id in sorted(raw_expressions):
            item = raw_expressions[expression_id]
            text = str(item.get("exp", "")).strip()
            if not text:
                raise ValueError(f"Empty target text for {source}/{video_id}/{expression_id}")
            anno_ids = (str(running_anno_id),) if synthetic_anno_ids else _string_tuple(item.get("anno_id"))
            if not anno_ids and skip_missing_mask_annotation_id:
                if excluded_missing_anno_ids is None:
                    raise ValueError("An exclusion audit list is required when skipping unlabeled expressions")
                excluded_missing_anno_ids.append(f"{source}/{video_id}/{expression_id}")
                continue
            if not anno_ids:
                raise ValueError(f"Missing mask annotation ID for {source}/{video_id}/{expression_id}")
            expressions.append(
                RefVOSExpression(
                    source=source,
                    video_id=str(video_id),
                    expression_id=str(expression_id),
                    text=text,
                    frame_names=frames,
                    mask_anno_ids=anno_ids,
                    object_ids=_string_tuple(item.get("obj_id", [0])),
                )
            )
            running_anno_id += 1
    return expressions


def sample_frame_indices(frame_count: int, count: int, rng: random.Random) -> tuple[int, ...]:
    """Match the official RefVOS frame sampling rule, including short clips."""
    if frame_count <= 0 or count <= 0:
        raise ValueError("frame_count and count must be positive")
    if frame_count > count + 1:
        indices = rng.sample(range(frame_count), count)
    else:
        indices = [rng.randrange(frame_count) for _ in range(count)]
    return tuple(sorted(indices))


def load_mask_dictionary(mask_path: str | Path) -> Mapping[str, Any]:
    """Load a trusted official annotation on a compute node, not a login node.

    Pickle executes code during loading. Pass only the known Sa2VA dataset file,
    never an untrusted or user-supplied arbitrary pickle.
    """
    path = Path(mask_path)
    if path.suffix == ".json":
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    elif path.suffix == ".pkl":
        with path.open("rb") as handle:
            data = pickle.load(handle)
    else:
        raise ValueError(f"Unsupported mask dictionary: {path}")
    if not isinstance(data, dict) or not data:
        raise ValueError(f"Invalid mask dictionary: {path}")
    return data


def load_refvos_sample(
    expression: RefVOSExpression,
    *,
    image_root: str | Path,
    mask_dict: Mapping[str, Any],
    frame_indices: tuple[int, ...],
) -> RefVOSSample:
    """Decode one expression; OR all its annotation masks on each chosen frame."""
    import numpy as np
    from PIL import Image
    from pycocotools import mask as mask_utils

    if not frame_indices or tuple(sorted(frame_indices)) != frame_indices:
        raise ValueError("frame_indices must be nonempty and sorted")
    if any(index < 0 or index >= len(expression.frame_names) for index in frame_indices):
        raise IndexError("Frame index is outside the expression's frame list")

    image_dir = Path(image_root) / expression.video_id
    images = []
    for index in frame_indices:
        image_path = image_dir / f"{expression.frame_names[index]}.jpg"
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        with Image.open(image_path) as image:
            images.append(image.convert("RGB"))
    height, width = images[0].height, images[0].width
    if any(image.size != (width, height) for image in images):
        raise ValueError(f"Inconsistent frame dimensions in {image_dir}")

    annotations = []
    for anno_id in expression.mask_anno_ids:
        if anno_id not in mask_dict:
            raise KeyError(f"Missing mask key {anno_id} for {expression.source}/{expression.video_id}/{expression.expression_id}")
        sequence = mask_dict[anno_id]
        if len(sequence) != len(expression.frame_names):
            raise ValueError(
                f"Mask/frame count mismatch for {expression.source}/{expression.video_id}/{expression.expression_id}/"
                f"{anno_id}: {len(sequence)} != {len(expression.frame_names)}"
            )
        annotations.append(sequence)

    masks = []
    for index in frame_indices:
        union = np.zeros((height, width), dtype=bool)
        for annotation in annotations:
            rle = annotation[index]
            if rle is None:
                continue  # Target can legitimately be invisible on a frame.
            decoded = mask_utils.decode(rle)
            if decoded.ndim == 3:
                decoded = decoded.any(axis=2)
            if decoded.shape != (height, width):
                raise ValueError(f"RLE/image shape mismatch: {decoded.shape} != {(height, width)}")
            union |= decoded.astype(bool)
        masks.append(union)
    return RefVOSSample(expression, frame_indices, images, np.stack(masks))
