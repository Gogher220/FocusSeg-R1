from __future__ import annotations

import os
from pathlib import Path

from transformers import AutoConfig

from configuration_sa2va_pro import Sa2VAProConfig
from modeling_sa2va_pro import Sa2VAProModel
from sa2va_checkpoint_utils import load_official_sa2va_segmentation_components
from sa2va_pro_utils import SEG_TOKEN, build_processor


def load_sa2va_pro_model(
    base_model_name_or_path: str = "Qwen/Qwen2-VL-2B-Instruct",
    config_path: str | None = None,
    segmentation_init_path: str | None = None,
    trust_remote_code: bool = True,
):
    load_path = config_path or base_model_name_or_path
    is_sa2va_checkpoint = False

    if config_path:
        base_config = AutoConfig.from_pretrained(load_path, trust_remote_code=trust_remote_code)
        is_sa2va_checkpoint = isinstance(base_config, Sa2VAProConfig) or getattr(base_config, "model_type", None) == Sa2VAProConfig.model_type
        config = base_config if isinstance(base_config, Sa2VAProConfig) else Sa2VAProConfig.from_dict(base_config.to_dict())
    else:
        base_config = AutoConfig.from_pretrained(load_path, trust_remote_code=trust_remote_code)
        is_sa2va_checkpoint = isinstance(base_config, Sa2VAProConfig) or getattr(base_config, "model_type", None) == Sa2VAProConfig.model_type
        if is_sa2va_checkpoint:
            config = Sa2VAProConfig.from_dict(base_config.to_dict())
        else:
            config = Sa2VAProConfig.from_dict(base_config.to_dict())
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

    if has_local_weights and is_sa2va_checkpoint:
        model = Sa2VAProModel.from_pretrained(load_path, config=config, trust_remote_code=trust_remote_code)
    else:
        model = Sa2VAProModel(config)
        current_embedding_rows = model.get_input_embeddings().weight.shape[0]
        if len(processor.tokenizer) > current_embedding_rows:
            model.resize_token_embeddings(len(processor.tokenizer))
    model.set_seg_token_id(seg_token_id)

    segmentation_init_path = segmentation_init_path or os.environ.get("SA2VA_PRO_SEGMENTATION_INIT")
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
