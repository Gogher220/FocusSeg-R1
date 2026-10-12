from __future__ import annotations

import os
from pathlib import Path

from transformers import AutoConfig

from configuration_sa2va_pro import Sa2VAProConfig
from modeling_sa2va_pro import Sa2VAProModel
from sa2va_checkpoint_utils import load_official_sa2va_segmentation_components
from sa2va_pro_utils import SEG_TOKEN, build_processor


def load_sa2va_pro_model(
    base_model_name_or_path: str,
    config_path: str | None = None,
    segmentation_init_path: str | None = None,
    trust_remote_code: bool = True,
):
    load_path = config_path or base_model_name_or_path
    base_config = AutoConfig.from_pretrained(load_path, trust_remote_code=trust_remote_code)
    if getattr(base_config, "model_type", None) == "sa2va_chat":
        raise ValueError(
            "The official Sa2VA checkpoint is a baseline model, not a Sa2VA-Pro "
            "checkpoint. Evaluate it with the official Sa2VA AutoModel path. "
            "For Sa2VA-Pro, pass the Video-R1 model as base_model_name_or_path "
            "and the official checkpoint as segmentation_init_path."
        )

    is_sa2va_checkpoint = (
        isinstance(base_config, Sa2VAProConfig)
        or getattr(base_config, "model_type", None) == Sa2VAProConfig.model_type
    )
    config = (
        base_config
        if isinstance(base_config, Sa2VAProConfig)
        else Sa2VAProConfig.from_dict(base_config.to_dict())
    )
    if not config_path and not is_sa2va_checkpoint:
        config.base_model_name_or_path = base_model_name_or_path

    if not getattr(config, "base_model_name_or_path", None):
        config.base_model_name_or_path = base_model_name_or_path
    config.seg_token = getattr(config, "seg_token", SEG_TOKEN)

    processor_source = load_path if Path(str(load_path)).exists() else config.base_model_name_or_path
    processor, added = build_processor(processor_source, seg_token=config.seg_token, trust_remote_code=trust_remote_code)
    seg_token_id = processor.tokenizer.convert_tokens_to_ids(config.seg_token)
    config.seg_token_id = seg_token_id

    load_path_obj = Path(str(load_path))
    has_local_weights = load_path_obj.exists() and (
        any(load_path_obj.glob("model.safetensors"))
        or any(load_path_obj.glob("model-*.safetensors"))
        or any(load_path_obj.glob("pytorch_model*.bin"))
    )
    segmentation_init_path = segmentation_init_path or os.environ.get("SA2VA_PRO_SEGMENTATION_INIT")
    if has_local_weights and is_sa2va_checkpoint and segmentation_init_path:
        raise ValueError(
            "A saved FocusSeg-R1 checkpoint must not be reinitialized from Sa2VA. "
            "Unset SA2VA_PRO_SEGMENTATION_INIT and omit segmentation_init_path "
            "when loading the saved checkpoint."
        )
    if not (has_local_weights and is_sa2va_checkpoint) and not segmentation_init_path:
        raise ValueError(
            "Sa2VA-Pro initialization requires the trained segmentation interface "
            "from an official Sa2VA checkpoint. Supply segmentation_init_path "
            "or SA2VA_PRO_SEGMENTATION_INIT; SAM2-only initialization would leave "
            "the [SEG] projection randomly initialized."
        )

    if has_local_weights and is_sa2va_checkpoint:
        model = Sa2VAProModel.from_pretrained(load_path, config=config, trust_remote_code=trust_remote_code)
    else:
        model = Sa2VAProModel(config)
        current_embedding_rows = model.get_input_embeddings().weight.shape[0]
        if len(processor.tokenizer) > current_embedding_rows:
            model.resize_token_embeddings(len(processor.tokenizer))
    model.set_seg_token_id(seg_token_id)
    model.model.requires_grad_(False)

    sam2_ckpt = os.environ.get("SA2VA_PRO_SAM2_CKPT")
    if sam2_ckpt is None and not (has_local_weights and is_sa2va_checkpoint):
        sam2_ckpt = getattr(model.config, "sam2_ckpt_path", None)
    if segmentation_init_path:
        report = load_official_sa2va_segmentation_components(
            model,
            segmentation_init_path,
            target_seg_token_id=seg_token_id,
            seg_token=config.seg_token,
        )
        print(f"Loaded official Sa2VA segmentation components: {report}", flush=True)
    elif sam2_ckpt:
        model.load_sam2_checkpoint(sam2_ckpt)

    return model, processor


def save_sa2va_pro_bundle(
    output_dir: str | Path,
    model,
    processor,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
