from __future__ import annotations

from transformers.utils import logging

try:
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
except ImportError:  # pragma: no cover
    from transformers import Qwen2VLConfig as Qwen2_5_VLConfig


logger = logging.get_logger(__name__)


class Sa2VAProConfig(Qwen2_5_VLConfig):
    model_type = "sa2va_pro"

    def __init__(
        self,
        template: str | None = None,
        llm_source: str = "video_r1",
        base_model_name_or_path: str = "Qwen/Qwen2-VL-2B-Instruct",
        seg_token: str = "[SEG]",
        seg_token_id: int | None = None,
        sam2_ckpt_path: str | None = None,
        seg_loss_weight: float = 1.0,
        seg_bce_weight: float = 1.0,
        seg_dice_weight: float = 1.0,
        training_mode: str = "grpo_seg",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.template = template
        self.llm_source = llm_source
        self.base_model_name_or_path = base_model_name_or_path
        self.seg_token = seg_token
        self.seg_token_id = seg_token_id
        self.sam2_ckpt_path = sam2_ckpt_path
        self.seg_loss_weight = seg_loss_weight
        self.seg_bce_weight = seg_bce_weight
        self.seg_dice_weight = seg_dice_weight
        self.training_mode = training_mode

    def to_dict(self):
        output = super().to_dict()
        output["template"] = self.template
        output["llm_source"] = self.llm_source
        output["base_model_name_or_path"] = self.base_model_name_or_path
        output["seg_token"] = self.seg_token
        output["seg_token_id"] = self.seg_token_id
        output["sam2_ckpt_path"] = self.sam2_ckpt_path
        output["seg_loss_weight"] = self.seg_loss_weight
        output["seg_bce_weight"] = self.seg_bce_weight
        output["seg_dice_weight"] = self.seg_dice_weight
        output["training_mode"] = self.training_mode
        return output
