from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from datasets import Dataset, DatasetDict, load_dataset
from trl import ModelConfig, SFTConfig, SFTTrainer, ScriptArguments, TrlParser, get_peft_config

from sa2va_pro_factory import load_sa2va_pro_model
from sa2va_training_utils import (
    prepare_model_for_checkpointing,
    print_trainable_parameters,
    set_trainable_by_patterns,
)
from sa2va_pro_utils import load_segmentation_targets, load_visual_inputs, resolve_path


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


@dataclass
class Sa2VAScriptArguments(ScriptArguments):
    data_root: str | None = None
    freeze_video_r1: bool = False
    trainable_param_patterns: str = "text_hidden_fcs,lora_,modules_to_save"


def disable_accelerate_deepspeed_probe():
    # We are not using DeepSpeed in this training entrypoint, but recent
    # accelerate versions still probe the installed package during Trainer
    # setup. On clusters without a visible CUDA toolkit this crashes before
    # training starts, even for plain DDP jobs.
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


def prepare_messages(example: dict[str, Any], data_root: str | None = None):
    data_type, visual_input = load_visual_inputs(example, data_root=data_root)
    if example.get("problem_type") == "multiple choice":
        question = example["problem"] + "Options:\n" + "\n".join(example.get("options", [])) + "\n"
    else:
        question = example["problem"]
    user_text = QUESTION_TEMPLATE.format(Question=question) + TYPE_TEMPLATE.get(
        example["problem_type"], TYPE_TEMPLATE["free-form"]
    )

    content = []
    if data_type == "image":
        content.append({"type": "image", "image": visual_input})
    else:
        for frame in visual_input[:5]:
            content.append({"type": "image", "image": frame})
    content.append({"type": "text", "text": user_text})

    process_text = example.get("process", "")
    solution_text = example.get("solution", "")
    assistant_text = (process_text + "\n" + solution_text).strip()
    return [
        {"role": "user", "content": content},
        {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]},
    ]


def collate_fn_builder(processor, data_root: str | None = None):
    def collate_fn(examples):
        texts = []
        image_batches = []
        target_masks = []
        for example in examples:
            messages = prepare_messages(example, data_root=data_root)
            prompt_text = processor.apply_chat_template(messages, tokenize=False)
            texts.append(prompt_text)
            images = []
            for item in messages[0]["content"]:
                if item["type"] == "image":
                    images.append(item["image"])
            image_batches.append(images if images else None)
            target_masks.append(load_segmentation_targets(example, data_root=data_root))

        flat_images = [
            image
            for images in image_batches
            if images
            for image in images
        ]
        flat_images = flat_images or None
        inputs = processor(text=texts, images=flat_images, return_tensors="pt", padding=True)
        inputs = dict(inputs)
        labels = inputs["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels
        if any(mask is not None for mask in target_masks):
            inputs["seg_images"] = image_batches
            inputs["target_masks"] = target_masks
        return inputs

    return collate_fn


def has_valid_visual_path(example: dict[str, Any], data_root: str | None = None) -> bool:
    path_value = example.get("path") or example.get("image_path") or example.get("video_path")
    if not path_value:
        return False
    resolved = resolve_path(path_value, data_root=data_root)
    return resolved is not None and resolved.exists()


def main(script_args, training_args, model_config):
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.ddp_find_unused_parameters = True
    if script_args.dataset_name.endswith(".json") or script_args.dataset_name.endswith(".jsonl"):
        dataset = DatasetDict({"train": Dataset.from_json(script_args.dataset_name)})
    else:
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    before_count = len(dataset["train"])
    dataset["train"] = dataset["train"].filter(
        has_valid_visual_path,
        fn_kwargs={"data_root": script_args.data_root},
        num_proc=1,
    )
    after_count = len(dataset["train"])
    print(f"Filtered train dataset by valid visual path: {before_count} -> {after_count}", flush=True)

    model, processor = load_sa2va_pro_model(model_config.model_name_or_path)
    disable_accelerate_deepspeed_probe()
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    peft_config = None
    if getattr(model_config, "use_peft", False):
        default_lora_targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        if not getattr(model_config, "lora_target_modules", None):
            model_config.lora_target_modules = default_lora_targets
        if not getattr(model_config, "lora_modules_to_save", None):
            model_config.lora_modules_to_save = ["text_hidden_fcs"]
        if getattr(model_config, "lora_bias", None) is None:
            model_config.lora_bias = "none"
        peft_config = get_peft_config(model_config)
        if peft_config is not None:
            if not getattr(peft_config, "target_modules", None):
                peft_config.target_modules = default_lora_targets
            if not getattr(peft_config, "modules_to_save", None):
                peft_config.modules_to_save = ["text_hidden_fcs"]
        print(f"Using PEFT config: {peft_config}", flush=True)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        data_collator=collate_fn_builder(processor, data_root=script_args.data_root),
        peft_config=peft_config,
    )
    prepare_model_for_checkpointing(trainer.model)
    if script_args.freeze_video_r1:
        patterns = set_trainable_by_patterns(trainer.model, script_args.trainable_param_patterns)
        print(f"Freeze Video-R1 backbone active. Trainable name patterns: {patterns}", flush=True)
    print_trainable_parameters(trainer.model)
    if training_args.resume_from_checkpoint is not None:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    else:
        trainer.train()
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    parser = TrlParser((Sa2VAScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()
    main(script_args, training_args, model_config)
