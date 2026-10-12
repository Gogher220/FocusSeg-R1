from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision.transforms.functional import to_pil_image
from transformers import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:  # pragma: no cover
    from transformers import Qwen2VLForConditionalGeneration as Qwen2_5_VLForConditionalGeneration
from transformers import Qwen2VLForConditionalGeneration

from qwen_vl_utils import process_vision_info

from configuration_sa2va_pro import Sa2VAProConfig
from sam2 import SAM2
from sa2va_pro_utils import SEG_TOKEN, bce_loss_from_logits, dice_loss_from_logits, mask_iou_from_logits, resize_gt_masks


@dataclass
class Sa2VAProSegmentationOutput:
    prediction: str
    prediction_masks: list[np.ndarray]
    pred_mask_logits: torch.Tensor | None = None
    raw_generation: Any | None = None
    seg_readout_mode: str = "none"


class DirectResize:
    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        img = to_pil_image(image, mode="RGB")
        return np.array(img.resize((self.target_length, self.target_length)))


class Sa2VAProModel(PreTrainedModel):
    config_class = Sa2VAProConfig
    main_input_name = "pixel_values"
    base_model_prefix = "language_model"
    _no_split_modules = ["Qwen2VLDecoderLayer", "SAM2"]
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True

    @staticmethod
    def _is_qwen2_5_base_model(base_model_name: str, config: PretrainedConfig) -> bool:
        model_type = getattr(config, "model_type", "") or ""
        arch = " ".join(getattr(config, "architectures", []) or [])
        base_name = str(base_model_name)
        return (
            "qwen2_5_vl" in model_type.lower()
            or "Qwen2_5" in arch
            or "Qwen2.5-VL" in base_name
        )

    def __init__(self, config, model=None, use_flash_attn: bool = False):
        super().__init__(config)
        self.extra_image_processor = DirectResize(target_length=1024)
        self.min_pixels = 512 * 28 * 28
        self.max_pixels = 2048 * 28 * 28
        self.torch_dtype = torch.bfloat16
        self.seg_token = getattr(config, "seg_token", SEG_TOKEN)
        self.seg_token_id = getattr(config, "seg_token_id", None)
        env_seg_loss_weight = os.environ.get("SA2VA_PRO_SEG_LOSS_WEIGHT")
        self.seg_loss_weight = float(env_seg_loss_weight if env_seg_loss_weight is not None else getattr(config, "seg_loss_weight", 1.0))
        self.seg_bce_weight = float(getattr(config, "seg_bce_weight", 1.0))
        self.seg_dice_weight = float(getattr(config, "seg_dice_weight", 1.0))
        self.sam2_ckpt_path = os.environ.get("SA2VA_PRO_SAM2_CKPT") or getattr(config, "sam2_ckpt_path", None)

        if model is not None:
            self.model = model
        else:
            base_model_name = getattr(config, "base_model_name_or_path", "Qwen/Qwen2-VL-2B-Instruct")
            base_model_cls = (
                Qwen2_5_VLForConditionalGeneration
                if self._is_qwen2_5_base_model(base_model_name, config)
                else Qwen2VLForConditionalGeneration
            )
            self.model = base_model_cls.from_pretrained(
                base_model_name,
                torch_dtype=self.torch_dtype,
                attn_implementation="flash_attention_2" if use_flash_attn else "eager",
            )

        text_config = getattr(config, "text_config", None)
        llm_hidden_size = getattr(text_config, "hidden_size", None)
        if llm_hidden_size is None:
            llm_hidden_size = getattr(config, "hidden_size", None)
        if llm_hidden_size is None:
            raise ValueError("Cannot determine the Video-R1 language hidden size")

        self.grounding_encoder = SAM2()
        out_dim = self.grounding_encoder.hidden_dim
        self.text_hidden_fcs = nn.Sequential(
            nn.Linear(llm_hidden_size, llm_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(llm_hidden_size, out_dim),
            nn.Dropout(0.0),
        )

    @property
    def lm_head(self):
        return self.model.lm_head

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.model.get_output_embeddings()

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.model.prepare_inputs_for_generation(*args, **kwargs)

    def enable_input_require_grads(self):
        if hasattr(self.model, "enable_input_require_grads"):
            return self.model.enable_input_require_grads()
        embeddings = self.get_input_embeddings()
        if embeddings is not None and hasattr(embeddings, "weight"):
            embeddings.weight.requires_grad_(True)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        checkpoint_kwargs = dict(gradient_checkpointing_kwargs or {})
        checkpoint_kwargs.setdefault("use_reentrant", False)

        self.enable_input_require_grads()
        if hasattr(self.model, "config") and hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False
        if hasattr(self.config, "use_cache"):
            self.config.use_cache = False

        if hasattr(self.model, "gradient_checkpointing_enable"):
            try:
                return self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=checkpoint_kwargs)
            except TypeError:
                try:
                    return self.model.gradient_checkpointing_enable(**checkpoint_kwargs)
                except TypeError:
                    return self.model.gradient_checkpointing_enable()

        try:
            return super().gradient_checkpointing_enable(gradient_checkpointing_kwargs=checkpoint_kwargs)
        except TypeError:
            return super().gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        if hasattr(self.model, "gradient_checkpointing_disable"):
            return self.model.gradient_checkpointing_disable()
        return super().gradient_checkpointing_disable()

    def resize_token_embeddings(self, new_num_tokens: int, pad_to_multiple_of: int | None = None, mean_resizing: bool = True):
        resized = self.model.resize_token_embeddings(
            new_num_tokens,
            pad_to_multiple_of=pad_to_multiple_of,
            mean_resizing=mean_resizing,
        )
        self.config.vocab_size = new_num_tokens
        if hasattr(self.config, "text_config") and hasattr(self.config.text_config, "vocab_size"):
            self.config.text_config.vocab_size = new_num_tokens
        return resized

    def set_seg_token_id(self, seg_token_id: int):
        self.seg_token_id = int(seg_token_id)
        self.config.seg_token_id = int(seg_token_id)

    def load_sam2_checkpoint(self, ckpt_path: str):
        self.grounding_encoder.load_checkpoint(ckpt_path)
        self.sam2_ckpt_path = ckpt_path
        self.config.sam2_ckpt_path = ckpt_path

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        seg_images=None,
        target_masks=None,
        **kwargs,
    ):
        wants_seg_loss = seg_images is not None and target_masks is not None
        output_hidden_states = kwargs.pop("output_hidden_states", None)
        return_dict = kwargs.pop("return_dict", True)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=output_hidden_states or wants_seg_loss,
            return_dict=return_dict,
            **kwargs,
        )

        if not wants_seg_loss:
            return outputs
        if not hasattr(outputs, "hidden_states") or not hasattr(outputs, "loss"):
            return outputs

        seg_loss_dict = self.compute_teacher_forced_segmentation_loss(
            input_ids=input_ids,
            hidden_states=outputs.hidden_states,
            seg_images=seg_images,
            target_masks=target_masks,
        )
        if not getattr(self, "_seg_sft_debug_printed", False):
            print(
                "Sa2VAPro segmentation SFT loss active: "
                f"seg_loss={seg_loss_dict['loss'].detach().float().item():.6f}, "
                f"bce={seg_loss_dict['bce_loss'].detach().float().item():.6f}, "
                f"dice={seg_loss_dict['dice_loss'].detach().float().item():.6f}, "
                f"mask_iou={seg_loss_dict['mask_iou'].detach().float().item():.6f}",
                flush=True,
            )
            self._seg_sft_debug_printed = True
        if outputs.loss is None:
            outputs.loss = seg_loss_dict["loss"]
        else:
            outputs.loss = outputs.loss + seg_loss_dict["loss"]
        return outputs

    def _decode_mask_logits_from_hidden_states(
        self,
        seg_hidden_states: torch.Tensor,
        g_pixel_values: torch.Tensor,
        ori_image_size: tuple[int, int],
        num_frames: int,
    ) -> torch.Tensor | None:
        if seg_hidden_states.numel() == 0:
            return None

        all_seg_hidden_states = self.text_hidden_fcs(seg_hidden_states)
        ret_logits: list[torch.Tensor] = []
        for seg_hidden_state in all_seg_hidden_states:
            seg_hidden_state = seg_hidden_state.unsqueeze(0)
            sam_states = self.grounding_encoder.get_sam2_embeddings(g_pixel_values)
            pred_masks = self.grounding_encoder.language_embd_inference(
                sam_states,
                [seg_hidden_state] * num_frames,
            )
            width, height = ori_image_size
            logits = F.interpolate(pred_masks, size=(height, width), mode="bilinear", align_corners=False)
            logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=20.0, neginf=-20.0)
            logits = logits.clamp(min=-20.0, max=20.0).to(pred_masks.dtype)
            logits = logits[:, 0]
            ret_logits.append(logits)
        return torch.stack(ret_logits, dim=0) if ret_logits else None

    def compute_teacher_forced_segmentation_loss(
        self,
        input_ids: torch.Tensor | None,
        hidden_states,
        seg_images,
        target_masks,
    ) -> dict[str, torch.Tensor]:
        if self.seg_token_id is None or input_ids is None or hidden_states is None:
            return self.compute_segmentation_loss(None, None)

        last_hidden_states = hidden_states[-1]
        if not isinstance(seg_images, (list, tuple)):
            seg_images = [seg_images]
        if not isinstance(target_masks, (list, tuple)):
            target_masks = [target_masks]

        per_example: dict[str, list[torch.Tensor]] = {
            "loss": [],
            "bce_loss": [],
            "dice_loss": [],
            "mask_iou": [],
        }

        batch_size = last_hidden_states.shape[0]
        for batch_idx in range(batch_size):
            if batch_idx >= len(seg_images) or batch_idx >= len(target_masks):
                continue
            visual = seg_images[batch_idx]
            masks = target_masks[batch_idx]
            if visual is None or masks is None:
                continue

            seg_hidden_states = get_seg_hidden_states(
                last_hidden_states[batch_idx],
                input_ids[batch_idx],
                seg_id=self.seg_token_id,
            )
            if seg_hidden_states.numel() == 0:
                continue

            if isinstance(visual, Image.Image):
                image = visual
                video = None
            else:
                image = None
                video = list(visual)
                if not video:
                    continue

            g_pixel_values, ori_image_size, num_frames, _ = self._prepare_grounding_inputs(image=image, video=video)
            g_pixel_values = g_pixel_values.to(self.device)
            pred_mask_logits = self._decode_mask_logits_from_hidden_states(
                seg_hidden_states,
                g_pixel_values,
                ori_image_size,
                num_frames,
            )
            seg_loss_dict = self.compute_segmentation_loss(pred_mask_logits, masks)
            for key in per_example:
                per_example[key].append(seg_loss_dict[key])

        if not per_example["loss"]:
            return self.compute_segmentation_loss(None, None)

        return {
            key: torch.stack(values).mean()
            for key, values in per_example.items()
        }

    def _prepare_grounding_inputs(
        self,
        image: Image.Image | None = None,
        video: list[Image.Image] | None = None,
    ):
        if video is not None:
            ori_image_size = video[0].size
            extra_pixel_values = []
            content = []
            for frame_idx, frame_image in enumerate(video):
                g_image = np.array(frame_image)
                g_image = self.extra_image_processor.apply_image(g_image)
                g_image = torch.from_numpy(g_image).permute(2, 0, 1).contiguous()
                extra_pixel_values.append(g_image)
                if frame_idx < 5:
                    content.append({"type": "image", "image": frame_image})
            g_pixel_values = torch.stack(
                [self.grounding_encoder.preprocess_image(pixel) for pixel in extra_pixel_values]
            ).to(self.torch_dtype)
            return g_pixel_values, ori_image_size, min(5, len(video)), content

        if image is None:
            raise ValueError("Either image or video must be provided for segmentation.")

        ori_image_size = image.size
        g_image = np.array(image)
        g_image = self.extra_image_processor.apply_image(g_image)
        g_pixel_values = torch.from_numpy(g_image).permute(2, 0, 1).contiguous().to(self.torch_dtype)
        g_pixel_values = torch.stack(
            [self.grounding_encoder.preprocess_image(g_pixel_values)]
        ).to(self.torch_dtype)
        return g_pixel_values, ori_image_size, 1, None

    def _decode_masks_from_generation(
        self,
        generation_output,
        mm_inputs,
        g_pixel_values: torch.Tensor,
        ori_image_size: tuple[int, int],
        num_frames: int,
    ) -> tuple[list[np.ndarray], torch.Tensor | None]:
        if self.seg_token_id is None:
            return [], None

        hidden_states = generation_output.hidden_states
        last_hidden_states = [item[-1][0] for item in hidden_states]
        last_hidden_states = torch.cat(last_hidden_states, dim=0)
        seg_hidden_states = get_seg_hidden_states(
            last_hidden_states,
            generation_output.sequences[0][:-1],
            seg_id=self.seg_token_id,
            start_index=mm_inputs.input_ids.shape[1],
        )
        if seg_hidden_states.numel() == 0:
            return [], None

        ret_masks: list[np.ndarray] = []
        stacked_logits = self._decode_mask_logits_from_hidden_states(
            seg_hidden_states,
            g_pixel_values,
            ori_image_size,
            num_frames,
        )
        if stacked_logits is None:
            return [], None
        for logits in stacked_logits:
            masks = (logits.sigmoid() > 0.5).cpu().numpy()
            ret_masks.append(masks)
        return ret_masks, stacked_logits

    def _forced_seg_hidden_states(self, generation_output, mm_inputs, processor) -> torch.Tensor:
        """Read a contextual [SEG] state when a frozen Video-R1 never generates it.

        Continue the generated answer with Sa2VA's short segmentation answer
        template. The continuation is teacher-forced; it does not update weights
        or pretend that Video-R1 generated a [SEG] token itself.
        """
        if self.seg_token_id is None:
            raise ValueError("[SEG] token id is not set")

        prompt_length = mm_inputs.input_ids.shape[1]
        sequence = generation_output.sequences[0].to(mm_inputs.input_ids.device)
        if sequence.numel() < prompt_length:
            raise ValueError("Generated sequence is shorter than the multimodal prompt")

        end_ids = self.model.generation_config.eos_token_id
        if isinstance(end_ids, int):
            end_ids = [end_ids]
        stop_ids = {int(token_id) for token_id in (end_ids or [])}
        pad_token_id = processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            stop_ids.add(int(pad_token_id))
        while sequence.numel() > prompt_length and int(sequence[-1]) in stop_ids:
            sequence = sequence[:-1]

        suffix_ids = processor.tokenizer.encode(
            f" It is {self.seg_token}.", add_special_tokens=False
        )
        seg_positions = [index for index, token_id in enumerate(suffix_ids)
                         if token_id == self.seg_token_id]
        if len(seg_positions) != 1:
            raise ValueError(
                f"Forced segmentation continuation must contain one [SEG] token; "
                f"found {len(seg_positions)}"
            )
        suffix = torch.tensor(suffix_ids, dtype=sequence.dtype, device=sequence.device)
        forced_ids = torch.cat((sequence, suffix)).unsqueeze(0)
        forward_inputs = dict(mm_inputs)
        forward_inputs["input_ids"] = forced_ids
        prompt_attention_mask = mm_inputs.get("attention_mask")
        if prompt_attention_mask is None:
            forward_inputs["attention_mask"] = torch.ones_like(forced_ids)
        else:
            continuation_attention = torch.ones(
                (1, forced_ids.shape[1] - prompt_length),
                dtype=prompt_attention_mask.dtype,
                device=prompt_attention_mask.device,
            )
            forward_inputs["attention_mask"] = torch.cat(
                (prompt_attention_mask, continuation_attention), dim=1
            )
        outputs = self.model(
            **forward_inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        if outputs.hidden_states is None:
            raise RuntimeError("Video-R1 did not return hidden states for forced [SEG]")
        seg_position = sequence.numel() + seg_positions[0]
        return outputs.hidden_states[-1][0, seg_position : seg_position + 1]

    def generate_with_masks(
        self,
        messages: list[dict[str, Any]],
        processor,
        image: Image.Image | None = None,
        video: list[Image.Image] | None = None,
        max_new_tokens: int = 2048,
        do_sample: bool = False,
    ) -> Sa2VAProSegmentationOutput:
        if processor is None:
            raise ValueError("processor is required")

        if self.seg_token_id is None:
            tokenizer = getattr(processor, "tokenizer", None)
            if tokenizer is not None:
                seg_id = tokenizer.convert_tokens_to_ids(self.seg_token)
                if seg_id != tokenizer.unk_token_id:
                    self.set_seg_token_id(seg_id)

        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        mm_inputs = processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        mm_inputs = mm_inputs.to(self.device)

        generation_output = self.model.generate(
            **mm_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
        generate_output_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(mm_inputs.input_ids, generation_output.sequences)
        ]
        prediction = processor.batch_decode(generate_output_trimmed, skip_special_tokens=False)[0].strip()

        prediction_masks: list[np.ndarray] = []
        pred_mask_logits: torch.Tensor | None = None
        seg_readout_mode = "none"
        if image is not None or video is not None:
            g_pixel_values, ori_image_size, num_frames, _ = self._prepare_grounding_inputs(image=image, video=video)
            g_pixel_values = g_pixel_values.to(self.device)
            prediction_masks, pred_mask_logits = self._decode_masks_from_generation(
                generation_output,
                mm_inputs,
                g_pixel_values,
                ori_image_size,
                num_frames,
            )
            if prediction_masks:
                seg_readout_mode = "generated"
            else:
                seg_hidden_states = self._forced_seg_hidden_states(
                    generation_output, mm_inputs, processor
                )
                pred_mask_logits = self._decode_mask_logits_from_hidden_states(
                    seg_hidden_states, g_pixel_values, ori_image_size, num_frames
                )
                if pred_mask_logits is not None:
                    prediction_masks = [
                        (logits.sigmoid() > 0.5).cpu().numpy()
                        for logits in pred_mask_logits
                    ]
                    seg_readout_mode = "forced_continuation"

        return Sa2VAProSegmentationOutput(
            prediction=prediction,
            prediction_masks=prediction_masks,
            pred_mask_logits=pred_mask_logits,
            raw_generation=generation_output,
            seg_readout_mode=seg_readout_mode,
        )

    def compute_segmentation_loss(
        self,
        pred_mask_logits: torch.Tensor | None,
        target_masks: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        if pred_mask_logits is None or target_masks is None:
            zero = torch.tensor(0.0, device=self.device)
            return {
                "loss": zero,
                "bce_loss": zero,
                "dice_loss": zero,
                "mask_iou": zero,
            }

        if pred_mask_logits.ndim == 4:
            if pred_mask_logits.shape[1] == 1:
                pred_mask_logits = pred_mask_logits[:, 0]
            else:
                pred_mask_logits = pred_mask_logits.reshape(-1, *pred_mask_logits.shape[-2:])
        elif pred_mask_logits.ndim != 3:
            raise ValueError(f"Unexpected pred mask shape: {pred_mask_logits.shape}")

        pred_mask_logits = torch.nan_to_num(pred_mask_logits.float(), nan=0.0, posinf=20.0, neginf=-20.0)
        pred_mask_logits = pred_mask_logits.clamp(min=-20.0, max=20.0)
        target_masks = target_masks.to(pred_mask_logits.device).float()
        target_masks = resize_gt_masks(target_masks, pred_mask_logits.shape[-2:])

        pred_count = pred_mask_logits.shape[0]
        target_count = target_masks.shape[0]
        if target_count < pred_count:
            repeats = int(np.ceil(pred_count / max(target_count, 1)))
            target_masks = target_masks.repeat(repeats, 1, 1)[:pred_count]
        elif target_count > pred_count:
            target_masks = target_masks[:pred_count]

        bce = bce_loss_from_logits(pred_mask_logits, target_masks)
        dice = dice_loss_from_logits(pred_mask_logits, target_masks)
        iou = mask_iou_from_logits(pred_mask_logits, target_masks)
        total = self.seg_bce_weight * bce + self.seg_dice_weight * dice
        return {
            "loss": total * self.seg_loss_weight,
            "bce_loss": bce,
            "dice_loss": dice,
            "mask_iou": iou,
        }

    def predict_forward(
        self,
        image=None,
        video=None,
        text=None,
        past_text="",
        mask_prompts=None,
        tokenizer=None,
        processor=None,
        max_new_tokens: int = 512,
        do_sample: bool = False,
    ):
        if processor is None:
            raise ValueError("processor is required")

        text = (text or "").replace("<image>", "")
        if image is None and video is None and "<image>" not in past_text:
            messages = [{"role": "user", "content": [{"type": "text", "text": past_text + text}]}]
            result = self.generate_with_masks(
                messages=messages, processor=processor,
                max_new_tokens=max_new_tokens, do_sample=do_sample,
            )
            return {"prediction": result.prediction, "prediction_masks": result.prediction_masks,
                    "seg_readout_mode": result.seg_readout_mode}

        if video is not None:
            content = [{"type": "image", "image": frame} for frame in video[:5]]
            content.append({"type": "text", "text": text})
            messages = [{"role": "user", "content": content}]
            result = self.generate_with_masks(
                messages=messages, processor=processor, video=video,
                max_new_tokens=max_new_tokens, do_sample=do_sample,
            )
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": text},
                    ],
                }
            ]
            result = self.generate_with_masks(
                messages=messages, processor=processor, image=image,
                max_new_tokens=max_new_tokens, do_sample=do_sample,
            )

        return {"prediction": result.prediction, "prediction_masks": result.prediction_masks,
                "seg_readout_mode": result.seg_readout_mode}


def get_seg_hidden_states(
    hidden_states: torch.Tensor,
    output_ids: torch.Tensor,
    seg_id: int,
    start_index: int = 0,
):
    n_hidden = len(hidden_states)
    n_ids = len(output_ids)
    offset = max(0, n_ids - n_hidden)
    output_ids = output_ids[offset:]
    start_index = max(0, int(start_index) - offset)
    seg_mask = output_ids == seg_id
    if start_index > 0:
        seg_mask[:start_index] = False
    n_out = len(seg_mask)
    if n_out == 0:
        return hidden_states[0:0]
    return hidden_states[-n_out:][seg_mask]
