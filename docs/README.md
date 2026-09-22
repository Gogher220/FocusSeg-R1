# Sa2VA_Pro

`Sa2VA_Pro` is a fused training/inference package that combines:

- `Sa2VA` visual grounding and segmentation via `SAM2`
- `Video R1` style multimodal reasoning prompts and GRPO training flow
- a custom bridge that lets image/video reasoning and segmentation coexist in one package

## Package Structure

```text
Sa2VA_Pro/
├── configuration_sa2va_pro.py
├── modeling_sa2va_pro.py
├── sam2.py
├── sa2va_pro_factory.py
├── sa2va_pro_utils.py
├── sa2va_pro.env.yml
├── setup_sa2va_pro.sh
├── qwen_vl_utils/
├── r1_v/
├── src/
│   └── open_r1/
│       ├── grpo.py
│       ├── sft_video.py
│       ├── segmentation_rewards.py
│       └── trainer/
│           ├── __init__.py
│           └── grpo_trainer.py
└── scripts/
    ├── run_sft_sa2va_pro.sh
    └── run_grpo_sa2va_pro.sh
```

## What Changed

### 1. LLM side

- The original `Sa2VA` Qwen-based language head is replaced with a `Qwen2.5-VL` base compatible with `Video R1` style reasoning prompts.
- `Sa2VAProModel` now exposes:
  - standard `forward(...)` for supervised fine-tuning
  - `generate_with_masks(...)` for joint text + segmentation inference
  - `compute_segmentation_loss(...)` for auxiliary mask supervision

### 2. Segmentation side

- The full `Sa2VA` `SAM2` implementation is restored into this package.
- `[SEG]` token support is explicitly added through tokenizer expansion.
- Generated `[SEG]` hidden states are decoded into masks through the SAM2 branch.

### 3. RL side

- A custom GRPO entrypoint is provided in `src/open_r1/grpo.py`.
- Reward registry now supports:
  - `accuracy`
  - `format`
  - `segmentation_iou`
  - `segmentation_presence`
- Trainer adds segmentation auxiliary loss on top of reward optimization.

## Expected Dataset Format

Each sample should at least contain:

```json
{
  "problem": "Segment the main object and explain what it is.",
  "problem_type": "free-form",
  "data_type": "image",
  "path": "/images/example.png",
  "solution": "<answer>cat</answer>",
  "process": "<think>...</think>",
  "mask_path": "/masks/example.npy"
}
```

Supported mask fields:

- `mask_path`
- `mask_paths`
- `mask`
- `masks`
- `segmentation_mask`
- `segmentation_masks`
- `target_mask`
- `target_masks`

For videos, use:

```json
{
  "data_type": "video",
  "path": "/videos/example.mp4",
  "nframes": 16
}
```

## Setup

```bash
cd Sa2VA_Pro
bash setup_sa2va_pro.sh sa2va_pro
```

## Train

SFT:

```bash
bash scripts/run_sft_sa2va_pro.sh
```

GRPO:

```bash
bash scripts/run_grpo_sa2va_pro.sh
```

## Important Limitations

- The original `Sa2VA` weight shards referenced by `model.safetensors.index.json` are not present in your provided `Sa2VA` folder. This package therefore assumes you will initialize from an available `Qwen2.5-VL` base and then fine-tune.
- The current GRPO path is a practical custom bridge, not the untouched upstream `Video R1` trainer.
- For segmentation to improve meaningfully, your dataset must include mask supervision or mask-based reward targets.
- The included `sam2.py` still depends on CUDA extensions during some post-processing paths.
