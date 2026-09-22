from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Optional

from datasets import Dataset, DatasetDict, load_dataset
from rouge_score import rouge_scorer
from trl import GRPOConfig, ModelConfig, ScriptArguments, TrlParser, get_peft_config

try:
    from .trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainerModified
    from .segmentation_rewards import segmentation_iou_reward, segmentation_presence_reward
except ImportError:  # pragma: no cover
    from trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainerModified
    from segmentation_rewards import segmentation_iou_reward, segmentation_presence_reward


@dataclass
class GRPOScriptArguments(ScriptArguments):
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "segmentation_iou", "segmentation_presence"],
        metadata={"help": "Reward names. Supports accuracy, segmentation_iou, and segmentation_presence."},
    )
    max_pixels: Optional[int] = field(default=12845056, metadata={"help": "Maximum number of pixels for the image"})
    min_pixels: Optional[int] = field(default=3136, metadata={"help": "Minimum number of pixels for the image"})
    temporal: Optional[bool] = field(default=True, metadata={"help": "Whether using temporal GRPO"})
    len_control: Optional[bool] = field(default=True, metadata={"help": "Whether using length reward"})
    temporal_reward_weight: Optional[float] = field(default=0.35, metadata={"help": "Reward bonus added to correct ordered samples when temporal contrast passes."})
    temporal_threshold_ratio: Optional[float] = field(default=0.85, metadata={"help": "Temporal contrast threshold. Temporal reward activates when ordered accuracy >= ratio * shuffled accuracy."})
    spatial: Optional[bool] = field(default=False, metadata={"help": "Whether to enable S-GRPO with random black-square spatial occlusion."})
    spatial_reward_weight: Optional[float] = field(default=0.35, metadata={"help": "Reward bonus added to correct ordered samples when spatial contrast passes."})
    spatial_keep_ratio: Optional[float] = field(default=0.65, metadata={"help": "Deprecated compatibility option; random-square S-GRPO uses spatial_occ_ratio_min/max."})
    spatial_occ_ratio_min: Optional[float] = field(default=0.10, metadata={"help": "Minimum black-square area ratio over the full frame/image."})
    spatial_occ_ratio_max: Optional[float] = field(default=0.30, metadata={"help": "Maximum black-square area ratio over the full frame/image."})
    spatial_occ_frame_ratio: Optional[float] = field(default=0.4, metadata={"help": "Fraction of video frames selected for random black-square occlusion."})
    spatial_threshold_ratio: Optional[float] = field(default=0.85, metadata={"help": "Spatial contrast threshold. Reward activates when ordered accuracy >= ratio * occluded accuracy."})
    data_root: Optional[str] = field(default=None, metadata={"help": "Dataset root for image/video/mask assets"})
    freeze_video_r1: bool = field(default=False, metadata={"help": "Freeze the Video-R1 backbone and train only selected small modules/adapters."})
    trainable_param_patterns: str = field(default="lora_", metadata={"help": "Comma-separated parameter-name substrings kept trainable when freeze_video_r1 is true."})


def disable_accelerate_deepspeed_probe():
    try:
        import accelerate.utils.other as accelerate_other

        accelerate_other.is_deepspeed_available = lambda: False
    except Exception:
        pass

    try:
        import accelerate.utils as accelerate_utils

        accelerate_utils.is_deepspeed_available = lambda: False
    except Exception:
        pass


def _extract_answer(text: str) -> str:
    pattern = r"<answer>\s*(.*?)\s*</answer>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def _normalize_number(num_str: str):
    try:
        return float(num_str.replace(",", ""))
    except Exception:
        return None


def _normalize_segmentation_answer(text: str) -> str:
    text = text.replace("[SEG]", " ")
    text = re.sub(r"\s+", " ", text.lower()).strip()
    return text


def accuracy_reward(completions, solution, **kwargs):
    question_type = kwargs["problem_type"][0]
    rewards = []
    for completion, sol in zip(completions, solution):
        content = completion[0]["content"]
        output_ans = _extract_answer(content)
        gt_ans = _extract_answer(sol)
        reward = 0.0
        if question_type == "multiple choice":
            reward = 1.0 if output_ans.strip() == gt_ans.strip() else 0.0
        elif question_type == "numerical":
            gt_number = _normalize_number(gt_ans)
            out_number = _normalize_number(output_ans)
            if gt_number is not None and out_number is not None:
                reward = 1.0 if round(gt_number, 2) == round(out_number, 2) else 0.0
        elif question_type == "free-form":
            score = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True).score(gt_ans, output_ans)
            reward = (score["rouge1"].fmeasure + score["rouge2"].fmeasure + score["rougeL"].fmeasure) / 3
        elif question_type == "regression":
            gt_number = _normalize_number(gt_ans)
            out_number = _normalize_number(output_ans)
            if gt_number is not None and out_number is not None:
                rel_diff = (abs(out_number - gt_number) + 1e-9) / (abs(gt_number) + 1e-9)
                reward = 1 - min(1.0, max(0.0, rel_diff))
        elif question_type == "OCR":
            reward = 1.0 if output_ans.strip() == gt_ans.strip() else 0.0
        elif question_type == "segmentation":
            gt_requires_seg = "[SEG]" in gt_ans
            out_has_seg = "[SEG]" in output_ans
            if gt_requires_seg and not out_has_seg:
                reward = 0.0
            else:
                gt_text = _normalize_segmentation_answer(gt_ans)
                out_text = _normalize_segmentation_answer(output_ans)
                if gt_text and out_text:
                    if gt_text == out_text:
                        reward = 1.0
                    else:
                        score = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True).score(gt_text, out_text)
                        reward = max(score["rouge1"].fmeasure, score["rougeL"].fmeasure)
                if out_has_seg:
                    reward = max(reward, 0.5)
        rewards.append(float(max(0.0, min(1.0, reward))))
    return rewards


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "segmentation_iou": segmentation_iou_reward,
    "segmentation_presence": segmentation_presence_reward,
}


QUESTION_TEMPLATE = (
    "{Question}\n"
    "Please think about this question as if you were a human pondering deeply. "
    "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', "
    "'let's break it down', etc, or other natural language thought expressions "
    "It's encouraged to include self-reflection or verification in the reasoning process. "
    "Provide your detailed reasoning between the <think> </think> tags, and then give your final answer between the "
    "<answer> </answer> tags."
)

TYPE_TEMPLATE = {
    "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
    "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
    "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
    "free-form": " Please provide your text answer within the <answer> </answer> tags.",
    "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
    "segmentation": " Please identify the referred object or region and provide the segmentation result within the <answer> </answer> tags.",
}


def make_conversation_image_and_video(example):
    if example.get("problem_type") == "multiple choice":
        question = example["problem"] + "Options:\n" + "\n".join(example.get("options", [])) + "\n"
    else:
        question = example["problem"]

    msg = {
        "prompt": [
            {
                "role": "user",
                "content": [
                    {"type": example["data_type"]},
                    {
                        "type": "text",
                        "text": QUESTION_TEMPLATE.format(Question=question)
                        + TYPE_TEMPLATE.get(example["problem_type"], TYPE_TEMPLATE["free-form"]),
                    },
                ],
            }
        ]
    }
    return msg


def main(script_args, training_args, model_args):
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.ddp_find_unused_parameters = True
    disable_accelerate_deepspeed_probe()
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    peft_config = None
    if getattr(model_args, "use_peft", False):
        default_lora_targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        if not getattr(model_args, "lora_target_modules", None):
            model_args.lora_target_modules = default_lora_targets
        if getattr(model_args, "lora_bias", None) is None:
            model_args.lora_bias = "none"
        peft_config = get_peft_config(model_args)
        if peft_config is not None and not getattr(peft_config, "target_modules", None):
            peft_config.target_modules = default_lora_targets
        print(f"Using PEFT config: {peft_config}", flush=True)

    if script_args.dataset_name.endswith(".json") or script_args.dataset_name.endswith(".jsonl"):
        dataset = DatasetDict({"train": Dataset.from_json(script_args.dataset_name)})
    else:
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    dataset = dataset.map(make_conversation_image_and_video)
    trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainerModified

    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        script_args=script_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=peft_config,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )
    from sa2va_training_utils import prepare_model_for_checkpointing

    prepare_model_for_checkpointing(trainer.model)

    if script_args.freeze_video_r1:
        from sa2va_training_utils import print_trainable_parameters, set_trainable_by_patterns

        patterns = set_trainable_by_patterns(trainer.model, script_args.trainable_param_patterns)
        print(f"Freeze Video-R1 backbone active. Trainable name patterns: {patterns}", flush=True)
        print_trainable_parameters(trainer.model)

    if script_args.data_root:
        os.environ["SA2VA_PRO_DATA_ROOT"] = script_args.data_root

    if training_args.resume_from_checkpoint is not None:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    else:
        trainer.train()

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
