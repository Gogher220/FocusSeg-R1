from __future__ import annotations

from typing import Any

import torch

from sa2va_pro_utils import load_segmentation_targets


def build_zero_rewards(length: int) -> list[float]:
    return [0.0 for _ in range(length)]


def segmentation_iou_reward(
    completions,
    prompts,
    pred_mask_ious=None,
    **kwargs,
):
    if pred_mask_ious is None:
        return build_zero_rewards(len(completions))

    rewards = []
    for i in range(len(completions)):
        iou = pred_mask_ious[i] if i < len(pred_mask_ious) else 0.0
        rewards.append(float(max(0.0, min(1.0, iou))))
    return rewards


def has_segmentation_target(example: dict[str, Any]) -> bool:
    return load_segmentation_targets(example) is not None


def segmentation_presence_reward(completions, prompts, pred_mask_counts=None, **kwargs):
    rewards = []
    for idx in range(len(completions)):
        count = pred_mask_counts[idx] if pred_mask_counts is not None and idx < len(pred_mask_counts) else 0
        content = completions[idx][0].get("content", "") if completions[idx] else ""
        has_seg_token = "[SEG]" in content
        rewards.append(1.0 if count > 0 or has_seg_token else 0.0)
    return rewards
