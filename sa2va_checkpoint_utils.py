from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from sam2 import load_state_dict_to_model


OFFICIAL_INPUT_EMBEDDING_KEY = "model.model.language_model.embed_tokens.weight"
OFFICIAL_LM_HEAD_KEY = "model.lm_head.weight"
OFFICIAL_GROUNDING_PREFIX = "grounding_encoder.sam2_model."
OFFICIAL_PROJECTION_PREFIX = "text_hidden_fcs."


def _read_weight_map(checkpoint_dir: Path) -> dict[str, str]:
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing sharded checkpoint index: {index_path}")
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Invalid weight_map in {index_path}")
    return {str(key): str(value) for key, value in weight_map.items()}


def _load_selected_tensors(
    checkpoint_dir: Path,
    weight_map: dict[str, str],
    keys: list[str],
) -> dict[str, torch.Tensor]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        shard_name = weight_map.get(key)
        if shard_name is None:
            raise KeyError(f"Official checkpoint does not contain {key!r}")
        grouped[shard_name].append(key)

    tensors: dict[str, torch.Tensor] = {}
    for shard_name, shard_keys in grouped.items():
        shard_path = checkpoint_dir / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(f"Missing checkpoint shard: {shard_path}")
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            missing = sorted(set(shard_keys) - available)
            if missing:
                raise KeyError(f"Keys missing from {shard_path.name}: {missing}")
            for key in shard_keys:
                tensors[key] = handle.get_tensor(key)
    return tensors


def _load_tensor_row(
    checkpoint_dir: Path,
    weight_map: dict[str, str],
    key: str,
    row_index: int,
) -> torch.Tensor:
    shard_name = weight_map.get(key)
    if shard_name is None:
        raise KeyError(f"Official checkpoint does not contain {key!r}")
    shard_path = checkpoint_dir / shard_name
    if not shard_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint shard: {shard_path}")
    with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
        tensor_slice = handle.get_slice(key)
        shape = tensor_slice.get_shape()
        if len(shape) != 2 or not 0 <= row_index < shape[0]:
            raise ValueError(f"Cannot read row {row_index} from {key} with shape {shape}")
        return tensor_slice[row_index : row_index + 1].clone()


def load_official_sa2va_segmentation_components(
    model,
    checkpoint_dir: str | Path,
    target_seg_token_id: int,
    seg_token: str = "[SEG]",
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, Any]:
    """Load only the trained Sa2VA segmentation interface into a Video-R1 model.

    The Video-R1/Qwen backbone remains untouched except for the single input
    embedding and LM-head rows associated with ``[SEG]``.
    """

    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    weight_map = _read_weight_map(checkpoint_dir)

    source_tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_dir,
        local_files_only=True,
        trust_remote_code=True,
        use_fast=False,
    )
    source_seg_token_id = source_tokenizer.convert_tokens_to_ids(seg_token)
    if source_seg_token_id is None or source_seg_token_id == source_tokenizer.unk_token_id:
        raise ValueError(f"Official checkpoint tokenizer does not contain {seg_token!r}")

    projection_keys = sorted(
        key for key in weight_map if key.startswith(OFFICIAL_PROJECTION_PREFIX)
    )
    expected_projection_keys = {
        f"{OFFICIAL_PROJECTION_PREFIX}{key}"
        for key in model.text_hidden_fcs.state_dict().keys()
    }
    if set(projection_keys) != expected_projection_keys:
        raise ValueError(
            "Projection keys do not match the local architecture: "
            f"missing={sorted(expected_projection_keys - set(projection_keys))}, "
            f"unexpected={sorted(set(projection_keys) - expected_projection_keys)}"
        )
    projection_tensors = _load_selected_tensors(
        checkpoint_dir, weight_map, projection_keys
    )
    projection_state = {
        key.removeprefix(OFFICIAL_PROJECTION_PREFIX): value
        for key, value in projection_tensors.items()
    }
    model.text_hidden_fcs.load_state_dict(projection_state, strict=True)

    grounding_keys = sorted(
        key for key in weight_map if key.startswith(OFFICIAL_GROUNDING_PREFIX)
    )
    if not grounding_keys:
        raise ValueError("Official checkpoint contains no grounding_encoder weights")
    grounding_tensors = _load_selected_tensors(
        checkpoint_dir, weight_map, grounding_keys
    )
    grounding_state = {
        key.removeprefix(OFFICIAL_GROUNDING_PREFIX): value
        for key, value in grounding_tensors.items()
    }
    load_state_dict_to_model(model.grounding_encoder.sam2_model, grounding_state)

    source_input_row = _load_tensor_row(
        checkpoint_dir,
        weight_map,
        OFFICIAL_INPUT_EMBEDDING_KEY,
        int(source_seg_token_id),
    )
    source_output_row = _load_tensor_row(
        checkpoint_dir,
        weight_map,
        OFFICIAL_LM_HEAD_KEY,
        int(source_seg_token_id),
    )

    input_embeddings = model.get_input_embeddings().weight
    output_module = model.get_output_embeddings()
    if output_module is None or not hasattr(output_module, "weight"):
        raise ValueError("The Video-R1 model does not expose LM-head weights")
    output_embeddings = output_module.weight

    if not 0 <= int(target_seg_token_id) < input_embeddings.shape[0]:
        raise ValueError(
            f"Target [SEG] id {target_seg_token_id} is outside input embeddings "
            f"with {input_embeddings.shape[0]} rows"
        )
    if not 0 <= int(target_seg_token_id) < output_embeddings.shape[0]:
        raise ValueError(
            f"Target [SEG] id {target_seg_token_id} is outside LM head "
            f"with {output_embeddings.shape[0]} rows"
        )
    if source_input_row.shape[1:] != input_embeddings[target_seg_token_id : target_seg_token_id + 1].shape[1:]:
        raise ValueError("Official and Video-R1 input embedding widths differ")
    if source_output_row.shape[1:] != output_embeddings[target_seg_token_id : target_seg_token_id + 1].shape[1:]:
        raise ValueError("Official and Video-R1 LM-head widths differ")

    with torch.no_grad():
        input_embeddings[target_seg_token_id].copy_(
            source_input_row[0].to(
                device=input_embeddings.device,
                dtype=input_embeddings.dtype,
            )
        )
        output_embeddings[target_seg_token_id].copy_(
            source_output_row[0].to(
                device=output_embeddings.device,
                dtype=output_embeddings.dtype,
            )
        )

    model.text_hidden_fcs.to(dtype=dtype)
    model.grounding_encoder.to(dtype=dtype)
    model.config.segmentation_init_path = str(checkpoint_dir)

    return {
        "checkpoint_dir": str(checkpoint_dir),
        "source_seg_token_id": int(source_seg_token_id),
        "target_seg_token_id": int(target_seg_token_id),
        "projection_tensor_count": len(projection_keys),
        "grounding_tensor_count": len(grounding_keys),
        "dtype": str(dtype),
    }
