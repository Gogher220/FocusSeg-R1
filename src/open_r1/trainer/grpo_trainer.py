from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

import torch
from datasets import Dataset, IterableDataset
from transformers import AutoProcessor, GenerationConfig, PreTrainedModel, TrainerCallback
from trl import GRPOConfig, GRPOTrainer
from trl.models import unwrap_model_for_generation

from modeling_sa2va_pro import Sa2VAProModel
from qwen_vl_utils import process_vision_info
from sa2va_pro_factory import load_sa2va_pro_model
from sa2va_pro_utils import (
    apply_random_spatial_occlusion_to_image,
    apply_random_spatial_occlusion_to_video,
    load_segmentation_targets,
    load_visual_inputs,
    resolve_path,
    shuffle_video_frames,
)


def _unwrap_sa2va_model(model) -> Sa2VAProModel | None:
    current = model
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        if isinstance(current, Sa2VAProModel):
            return current
        visited.add(id(current))
        current = getattr(current, "module", None) or getattr(current, "model", None) or getattr(current, "base_model", None)
    return None


class Qwen2VLGRPOTrainer(GRPOTrainer):
    def __init__(
        self,
        model,
        reward_funcs,
        args: GRPOConfig = None,
        script_args=None,
        train_dataset: Optional[Dataset | IterableDataset] = None,
        eval_dataset: Optional[Dataset | IterableDataset] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
        **kwargs,
    ):
        self.script_args = script_args
        self.data_root = getattr(script_args, "data_root", None)

        if isinstance(model, str):
            self.model_name_or_path = model
            sa2va_model, processor = load_sa2va_pro_model(model)
            model = sa2va_model
            self.processing_class = processor
        else:
            self.model_name_or_path = getattr(model.config, "_name_or_path", "Sa2VA_Pro")
            self.processing_class = AutoProcessor.from_pretrained(self.model_name_or_path, trust_remote_code=True)

        if hasattr(self.processing_class, "image_processor"):
            self.processing_class.image_processor.max_pixels = max_pixels
            self.processing_class.image_processor.min_pixels = min_pixels

        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.processing_class,
            callbacks=callbacks,
            **kwargs,
        )

        tokenizer = getattr(self.processing_class, "tokenizer", None)
        pad_token_id = getattr(self.processing_class, "pad_token_id", None)
        if pad_token_id is None and tokenizer is not None:
            pad_token_id = tokenizer.pad_token_id
        eos_token_id = getattr(self.processing_class, "eos_token_id", None)
        if eos_token_id is None and tokenizer is not None:
            eos_token_id = tokenizer.eos_token_id

        self.pad_token_id = int(pad_token_id)
        self.eos_token_id = int(eos_token_id)
        self.temporal = bool(getattr(script_args, "temporal", False))
        self.temporal_reward_weight = float(getattr(script_args, "temporal_reward_weight", 0.35))
        self.temporal_threshold_ratio = float(getattr(script_args, "temporal_threshold_ratio", 0.85))
        self.len_control = bool(getattr(script_args, "len_control", False))
        self.spatial = bool(getattr(script_args, "spatial", False))
        self.spatial_reward_weight = float(getattr(script_args, "spatial_reward_weight", 0.35))
        self.spatial_occ_ratio_min = float(getattr(script_args, "spatial_occ_ratio_min", 0.10))
        self.spatial_occ_ratio_max = float(getattr(script_args, "spatial_occ_ratio_max", 0.30))
        self.spatial_occ_frame_ratio = float(getattr(script_args, "spatial_occ_frame_ratio", 0.4))
        self.spatial_threshold_ratio = float(getattr(script_args, "spatial_threshold_ratio", 0.85))

        self.max_prompt_length = getattr(self, "max_prompt_length", getattr(args, "max_prompt_length", None))
        self.max_completion_length = getattr(self, "max_completion_length", getattr(args, "max_completion_length", 768))
        self.num_generations = getattr(self, "num_generations", getattr(args, "num_generations", 4))
        self.beta = getattr(self, "beta", getattr(args, "beta", 0.03))
        self.contrast_num_generations = max(1, self.num_generations // 2)

        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            top_p=0.95,
            temperature=1.0,
            num_return_sequences=self.num_generations,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
        )
        self.contrast_generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            top_p=0.95,
            temperature=1.0,
            num_return_sequences=self.contrast_num_generations,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
        )

        if not hasattr(self, "_metrics"):
            self._metrics = defaultdict(list)

    def _prepare_inputs(self, inputs):
        return inputs

    def _resolve_visual_source(self, example: dict[str, Any]):
        path_value = example.get("path") or example.get("image_path") or example.get("video_path")
        resolved = resolve_path(path_value, self.data_root)
        return str(resolved) if resolved is not None else None

    def _build_messages(self, example: dict[str, Any], visual_source):
        content: list[dict[str, Any]] = []
        if example.get("data_type") == "image":
            content.append({"type": "image", "image": visual_source})
        else:
            content.append({"type": "video", "video": visual_source})
        content.append(example["prompt"][0]["content"][1])
        return [{"role": "user", "content": content}]

    def _build_prompt_bundle(self, messages, device: torch.device):
        prompt_text = self.processing_class.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        prompt_inputs = self.processing_class(
            text=[prompt_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            min_pixels=self.processing_class.image_processor.min_pixels,
            max_pixels=self.processing_class.image_processor.max_pixels,
        )
        prompt_inputs = prompt_inputs.to(device)
        if self.max_prompt_length is not None:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -self.max_prompt_length :]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -self.max_prompt_length :]
        return {
            "prompt_text": prompt_text,
            "image_inputs": image_inputs,
            "video_inputs": video_inputs,
            "prompt_inputs": prompt_inputs,
        }

    def _build_prompt_inputs_from_vision(self, prompt_text, image_inputs, video_inputs, device: torch.device):
        prompt_inputs = self.processing_class(
            text=[prompt_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            min_pixels=self.processing_class.image_processor.min_pixels,
            max_pixels=self.processing_class.image_processor.max_pixels,
        )
        prompt_inputs = prompt_inputs.to(device)
        if self.max_prompt_length is not None:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -self.max_prompt_length :]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -self.max_prompt_length :]
        return prompt_inputs

    def _build_mm_kwargs_for_logps(self, prompt_inputs, repeats: int):
        mm_kwargs: dict[str, torch.Tensor] = {}
        for key, value in prompt_inputs.items():
            if key in {"input_ids", "attention_mask"}:
                continue
            if torch.is_tensor(value) and value.ndim > 0:
                mm_kwargs[key] = value.repeat_interleave(repeats, dim=0)
            else:
                mm_kwargs[key] = value
        return mm_kwargs

    def _get_per_token_logps(self, model, input_ids, **kwargs):
        logits = model(input_ids=input_ids, **kwargs).logits
        logits = logits[:, :-1, :]
        input_ids = input_ids[:, 1:]
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)

    def _decode_completions(self, completion_ids):
        completion_texts = self.processing_class.batch_decode(completion_ids, skip_special_tokens=False)
        completions = [[{"role": "assistant", "content": text}] for text in completion_texts]
        return completion_texts, completions

    def _build_reward_kwargs(self, inputs, repeats: int):
        reward_kwargs = {key: [] for key in inputs[0].keys() if key not in {"prompt", "completion"}}
        for key in reward_kwargs:
            for example in inputs:
                reward_kwargs[key].extend([example[key]] * repeats)
        return reward_kwargs

    def _get_segmentation_bundle(self, model, example, messages, visual_input, target_masks):
        sa2va_model = _unwrap_sa2va_model(model)
        if sa2va_model is None:
            return {
                "loss": torch.tensor(0.0, device=self.accelerator.device),
                "bce_loss": torch.tensor(0.0, device=self.accelerator.device),
                "dice_loss": torch.tensor(0.0, device=self.accelerator.device),
                "mask_iou": torch.tensor(0.0, device=self.accelerator.device),
                "pred_mask_count": 0,
            }

        generation = sa2va_model.generate_with_masks(
            messages=messages,
            processor=self.processing_class,
            image=visual_input if example.get("data_type") == "image" else None,
            video=visual_input if example.get("data_type") == "video" else None,
            max_new_tokens=self.max_completion_length,
            do_sample=True,
        )
        seg_loss_dict = sa2va_model.compute_segmentation_loss(generation.pred_mask_logits, target_masks)
        return {
            "loss": seg_loss_dict["loss"],
            "bce_loss": seg_loss_dict["bce_loss"],
            "dice_loss": seg_loss_dict["dice_loss"],
            "mask_iou": seg_loss_dict["mask_iou"],
            "pred_mask_count": len(generation.prediction_masks),
        }

    def _compute_rewards_per_func(
        self,
        inputs,
        repeated_prompts,
        completions,
        reward_kwargs,
        seg_bundle,
        device: torch.device,
    ):
        reward_kwargs = dict(reward_kwargs)
        if seg_bundle is not None:
            mask_iou = seg_bundle.get("mask_iou", 0.0)
            if torch.is_tensor(mask_iou):
                mask_iou = float(mask_iou.detach().float().cpu().item())
            pred_mask_count = int(seg_bundle.get("pred_mask_count", 0))
            reward_kwargs["pred_mask_ious"] = [mask_iou for _ in completions]
            reward_kwargs["pred_mask_counts"] = [pred_mask_count for _ in completions]
        rewards_per_func = torch.zeros(len(completions), len(self.reward_funcs), dtype=torch.float32, device=device)
        for idx, reward_func in enumerate(self.reward_funcs):
            reward = reward_func(prompts=repeated_prompts, completions=completions, **reward_kwargs)
            rewards_per_func[:, idx] = torch.tensor(reward, dtype=torch.float32, device=device)
        return rewards_per_func

    def _accuracy_index(self):
        for idx, reward_func in enumerate(self.reward_funcs):
            if getattr(reward_func, "__name__", "") == "accuracy_reward":
                return idx
        return 0

    def _correctness_mask(self, rewards_per_func: torch.Tensor):
        accuracy_scores = rewards_per_func[:, self._accuracy_index()]
        return accuracy_scores > 0.1

    def _compute_branch_outputs(self, generator_model, prompt_inputs, generation_config):
        prompt_completion_ids = generator_model.generate(**prompt_inputs, generation_config=generation_config)
        prompt_length = prompt_inputs["input_ids"].size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]
        is_eos = completion_ids == self.eos_token_id
        eos_idx = torch.full(
            (completion_ids.size(0),),
            completion_ids.size(1),
            dtype=torch.long,
            device=completion_ids.device,
        )
        if is_eos.any():
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(completion_ids.size(1), device=completion_ids.device).expand(completion_ids.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        return prompt_completion_ids, completion_ids, completion_mask, prompt_length

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The custom Sa2VAPro GRPO trainer does not support return_outputs.")
        if len(inputs) != 1:
            raise ValueError("This trainer currently supports per-device train batch size 1 only.")

        example = inputs[0]
        device = self.accelerator.device
        prompts = [example["prompt"]]
        data_type, visual_input = load_visual_inputs(example, self.data_root)
        target_masks = load_segmentation_targets(example, self.data_root)
        ordered_visual_source = self._resolve_visual_source(example) or visual_input
        ordered_messages = self._build_messages(example, ordered_visual_source)
        ordered_bundle = self._build_prompt_bundle(ordered_messages, device)
        ordered_prompt_inputs = ordered_bundle["prompt_inputs"]

        shuffled_messages = None
        shuffled_prompt_inputs = None
        if self.temporal and data_type == "video" and ordered_bundle["video_inputs"]:
            ordered_video = ordered_bundle["video_inputs"][0]
            if torch.is_tensor(ordered_video):
                indices = torch.randperm(ordered_video.size(0))
                shuffled_video_inputs = [ordered_video[indices]]
            else:
                shuffled_frames = shuffle_video_frames(list(ordered_video))
                shuffled_video_inputs = [shuffled_frames]
            shuffled_prompt_inputs = self._build_prompt_inputs_from_vision(
                prompt_text=ordered_bundle["prompt_text"],
                image_inputs=ordered_bundle["image_inputs"],
                video_inputs=shuffled_video_inputs,
                device=device,
            )

        spatial_meta: dict[str, Any] = {"applied": False}
        masked_messages = None
        masked_prompt_inputs = None
        if self.spatial:
            if data_type == "video":
                masked_visual_input, spatial_meta = apply_random_spatial_occlusion_to_video(
                    visual_input,
                    occ_ratio_min=self.spatial_occ_ratio_min,
                    occ_ratio_max=self.spatial_occ_ratio_max,
                    occ_frame_ratio=self.spatial_occ_frame_ratio,
                )
            elif data_type == "image":
                masked_visual_input, spatial_meta = apply_random_spatial_occlusion_to_image(
                    visual_input,
                    occ_ratio_min=self.spatial_occ_ratio_min,
                    occ_ratio_max=self.spatial_occ_ratio_max,
                )
            if spatial_meta.get("applied", False):
                masked_messages = self._build_messages(example, masked_visual_input)
                masked_bundle = self._build_prompt_bundle(masked_messages, device)
                masked_prompt_inputs = masked_bundle["prompt_inputs"]

        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            ordered_prompt_completion_ids, completion_ids, completion_mask, prompt_length = self._compute_branch_outputs(
                unwrapped_model,
                ordered_prompt_inputs,
                self.generation_config,
            )

            shuffled_completion_ids = None
            if shuffled_prompt_inputs is not None:
                _, shuffled_completion_ids, _, _ = self._compute_branch_outputs(
                    unwrapped_model,
                    shuffled_prompt_inputs,
                    self.contrast_generation_config,
                )

            masked_completion_ids = None
            if masked_prompt_inputs is not None:
                _, masked_completion_ids, _, _ = self._compute_branch_outputs(
                    unwrapped_model,
                    masked_prompt_inputs,
                    self.contrast_generation_config,
                )

        mm_kwargs = self._build_mm_kwargs_for_logps(ordered_prompt_inputs, self.num_generations)
        per_token_logps = self._get_per_token_logps(model, ordered_prompt_completion_ids, **mm_kwargs)
        per_token_logps = per_token_logps[:, prompt_length - 1 :]

        with torch.inference_mode():
            if getattr(self, "ref_model", None) is not None:
                ref_per_token_logps = self._get_per_token_logps(self.ref_model, ordered_prompt_completion_ids, **mm_kwargs)
            else:
                ref_per_token_logps = per_token_logps.detach()
        ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 :]

        x_clamped = torch.clamp(ref_per_token_logps - per_token_logps, min=-10, max=10)
        per_token_kl = torch.exp(x_clamped) - x_clamped - 1

        _, ordered_completions = self._decode_completions(completion_ids)
        ordered_reward_kwargs = self._build_reward_kwargs(inputs, self.num_generations)
        ordered_repeated_prompts = prompts * self.num_generations

        unwrapped_policy_model = self.accelerator.unwrap_model(model)
        seg_bundle = self._get_segmentation_bundle(unwrapped_policy_model, example, ordered_messages, visual_input, target_masks)
        rewards_per_func = self._compute_rewards_per_func(
            inputs=inputs,
            repeated_prompts=ordered_repeated_prompts,
            completions=ordered_completions,
            reward_kwargs=ordered_reward_kwargs,
            seg_bundle=seg_bundle,
            device=device,
        )

        shuffled_rewards_per_func = None
        if shuffled_completion_ids is not None:
            _, shuffled_completions = self._decode_completions(shuffled_completion_ids)
            shuffled_reward_kwargs = self._build_reward_kwargs(inputs, self.contrast_num_generations)
            shuffled_rewards_per_func = self._compute_rewards_per_func(
                inputs=inputs,
                repeated_prompts=prompts * self.contrast_num_generations,
                completions=shuffled_completions,
                reward_kwargs=shuffled_reward_kwargs,
                seg_bundle=seg_bundle,
                device=device,
            )

        masked_rewards_per_func = None
        if masked_completion_ids is not None:
            _, masked_completions = self._decode_completions(masked_completion_ids)
            masked_reward_kwargs = self._build_reward_kwargs(inputs, self.contrast_num_generations)
            masked_rewards_per_func = self._compute_rewards_per_func(
                inputs=inputs,
                repeated_prompts=prompts * self.contrast_num_generations,
                completions=masked_completions,
                reward_kwargs=masked_reward_kwargs,
                seg_bundle=seg_bundle,
                device=device,
            )

        rewards = rewards_per_func.sum(dim=1)
        ordered_correct_mask = self._correctness_mask(rewards_per_func)
        ordered_correct_ratio = ordered_correct_mask.float().mean()

        temporal_reward_scalar = torch.tensor(0.0, device=device)
        shuffled_correct_ratio = torch.tensor(0.0, device=device)
        if shuffled_rewards_per_func is not None:
            shuffled_correct_mask = self._correctness_mask(shuffled_rewards_per_func)
            shuffled_correct_ratio = shuffled_correct_mask.float().mean()
            if ordered_correct_mask.any() and ordered_correct_ratio >= self.temporal_threshold_ratio * shuffled_correct_ratio:
                rewards = rewards.clone()
                rewards[ordered_correct_mask] = rewards[ordered_correct_mask] + self.temporal_reward_weight
                temporal_reward_scalar = torch.tensor(self.temporal_reward_weight, device=device)

        spatial_reward_scalar = torch.tensor(0.0, device=device)
        masked_correct_ratio = torch.tensor(0.0, device=device)
        if masked_rewards_per_func is not None:
            masked_correct_mask = self._correctness_mask(masked_rewards_per_func)
            masked_correct_ratio = masked_correct_mask.float().mean()
            if ordered_correct_mask.any() and ordered_correct_ratio >= self.spatial_threshold_ratio * masked_correct_ratio:
                rewards = rewards.clone()
                rewards[ordered_correct_mask] = rewards[ordered_correct_mask] + self.spatial_reward_weight
                spatial_reward_scalar = torch.tensor(self.spatial_reward_weight, device=device)

        if self.len_control:
            selected_indices = torch.nonzero(ordered_correct_mask, as_tuple=True)[0].tolist()
            lengths = completion_mask.sum(dim=1)
            if len(selected_indices) > 1:
                for idx in selected_indices:
                    if 320 <= int(lengths[idx].item()) <= 512:
                        rewards[idx] += 0.2

        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1, unbiased=False)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        denom = completion_mask.sum(dim=1).clamp_min(1)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / denom).mean()

        seg_loss = seg_bundle["loss"]
        if torch.is_tensor(seg_loss):
            loss = loss + seg_loss

        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(dim=1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(dim=0)
        for idx, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_name = getattr(reward_func, "__name__", reward_func.__class__.__name__)
            self._metrics[f"rewards/{reward_name}"].append(float(reward_per_func[idx].item()))

        self._metrics["reward"].append(float(self.accelerator.gather_for_metrics(rewards).mean().item()))
        self._metrics["reward_std"].append(float(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item()))
        self._metrics["ordered_correct_ratio"].append(float(self.accelerator.gather_for_metrics(ordered_correct_ratio).mean().item()))
        self._metrics["shuffled_correct_ratio"].append(float(self.accelerator.gather_for_metrics(shuffled_correct_ratio).mean().item()))
        self._metrics["masked_correct_ratio"].append(float(self.accelerator.gather_for_metrics(masked_correct_ratio).mean().item()))
        self._metrics["temporal_rewards"].append(float(self.accelerator.gather_for_metrics(temporal_reward_scalar).mean().item()))
        self._metrics["spatial_rewards"].append(float(self.accelerator.gather_for_metrics(spatial_reward_scalar).mean().item()))
        self._metrics["spatial_mask_applied"].append(1.0 if spatial_meta.get("applied", False) else 0.0)

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / denom).mean()
        self._metrics["kl"].append(float(self.accelerator.gather_for_metrics(mean_kl).mean().item()))

        if torch.is_tensor(seg_bundle["bce_loss"]):
            self._metrics["seg_bce"].append(float(seg_bundle["bce_loss"].detach().cpu().item()))
        if torch.is_tensor(seg_bundle["dice_loss"]):
            self._metrics["seg_dice"].append(float(seg_bundle["dice_loss"].detach().cpu().item()))
        if torch.is_tensor(seg_bundle["mask_iou"]):
            self._metrics["seg_iou"].append(float(seg_bundle["mask_iou"].detach().cpu().item()))
        self._metrics["pred_mask_count"].append(float(seg_bundle["pred_mask_count"]))

        return loss


class Qwen2VLGRPOVLLMTrainerModified(Qwen2VLGRPOTrainer):
    pass
