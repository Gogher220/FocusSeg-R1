# FocusSeg-R1 / Sa2VA-Pro

Current experiment: **SEG-token supervised LoRA for referring image and video segmentation**, using Video-R1-7B and segmentation components initialized from Sa2VA-Qwen2.5-VL-7B. This repository contains code and configuration only; pretrained weights, trained checkpoints, datasets, and generated experiment artifacts are not included.

The current method trains language-model LoRA adapters, task-specific `[SEG]` input/output embedding rows, and the language-to-SAM2 projection. The original Video-R1 weights, visual encoder, and SAM2 remain frozen. Training combines assistant-token cross-entropy with point-sampled mask BCE and Dice losses. Segmentation inference must generate `[SEG]` itself.

Read the [current method and execution guide](docs/METHOD_SEG_LORA.md) for the complete training, validation, checkpoint, and resume procedure. The earlier [package documentation](docs/README.md) describes legacy SFT/GRPO entry points; those are not the entry points for this experiment. The current run does not use GRPO or an independent query module.

## Current code

| Purpose | Files |
| --- | --- |
| Model, task routing, losses, inference | `seg_lora/model.py` |
| Complete seven-source training index | `seg_lora/data.py` |
| Validation and provenance | `seg_lora/runtime.py` |
| Distributed preparation, smoke test, training | `local_scripts/seg_lora_run.py` |
| Atomic checkpoint saves and latest/best retention | `local_scripts/seg_lora_storage.py` |
| Shared data and sampling utilities | `focusseg_r1/full_data.py`, `refvos_data.py`, `full_runtime.py` |
| Four-GPU, three-epoch recipe | `configs/seg_lora_fresh_3epoch.json` |
| Cluster launchers | `scripts/yhbatch_seg_lora_*.sh` |

## Training recipe

Seven segmentation sources: RefCOCO, RefCOCO+, RefCOCOg, Ref-YouTube-VOS, MeViS, ReVOS, and Ref-SAV. The indexed run contains 459,052 expressions, including 114 explicitly empty ReVOS targets and the complete 72,509-expression Ref-SAV training split. This is not a reproduction of Sa2VA's full multitask data mixture.

Four GPUs, one expression per GPU per microbatch, eight accumulation steps, effective batch size 32, and three epochs give 43,038 optimizer updates. Checkpoints retain only the latest and best states after successful saves. Full Ref-SAV validation (1,945 expressions) runs every 2,000 updates and at epoch boundaries.

## Environment and assets

The reported training environment is Python 3.10, PyTorch 2.5.1+cu124, PEFT 0.19.1, and Transformers 4.57.6. Existing environment/setup files are historical starting points; the remote run used these versions. Obtain Video-R1, Sa2VA, and datasets separately under their upstream licenses and access conditions.

The committed JSON and cluster launchers retain the original run's absolute paths for provenance. Adapt paths and scheduler settings before a fresh run on another machine. An existing run additionally checks code/configuration/index identities when resuming; do not rewrite them underneath a running experiment.
