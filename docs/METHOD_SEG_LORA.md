# SEG-token LoRA method and execution guide

This document describes the current `seg_lora_fresh_v1` implementation. It supersedes the older query-module experiments as the active method. It does not claim that the planned three-epoch run has finished.

## Architecture and supervision

1. `SegLoRA.inputs` combines the referring expression with an image or up to five sampled video frames. The supervised assistant answer is `It is [SEG].`; the user prefix is excluded from language cross-entropy.
2. Language-decoder `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj` layers receive LoRA adapters: rank 128, alpha 256, dropout 0.05. Visual layers are excluded.
3. The `[SEG]` input embedding and output-head row are task-local trainable parameters, initialized from the official Sa2VA segmentation checkpoint. The original Video-R1 rows remain available for the QA route.
4. The last hidden state at the supervised `[SEG]` position passes through the trainable Sa2VA-initialized projection into SAM2. SAM2 parameters are frozen, while mask-loss gradients propagate through the decoder to the projection and LoRA path.
5. The loss is `assistant CE + 2 * point BCE + 0.5 * point Dice`. Each mask uses 12,544 sampled points, oversampling ratio 3.0, and uncertainty-sampling fraction 0.75. Explicit negative targets contribute all-zero masks.

`SegLoRA.predict` generates the answer autoregressively. Exactly one generated `[SEG]` is required; otherwise the validation records a generation failure and scores an empty prediction. A second forward pass on the generated sequence retrieves the hidden state at the token itself. The first five frames provide language prompts, and SAM2 propagates masks through the full input video.

The shared `Sa2VAProModel` still contains legacy inference helpers, including a forced-token fallback. The current SEG LoRA inference uses `SegLoRA.predict`, not that legacy fallback. Current training does not instantiate a query module or run GRPO.

`qa_route()` disables LoRA and the task-specific token rows. The smoke test checks unchanged QA logits for its tested input; this is not a general QA benchmark or a claim of unchanged performance on every task.

## Data

| Training source | Indexed expressions |
| --- | ---: |
| RefCOCO (UNC) | 120,624 |
| RefCOCO+ (UNC) | 120,191 |
| RefCOCOg (UMD) | 80,512 |
| Ref-YouTube-VOS | 12,913 |
| MeViS | 23,051 |
| ReVOS | 29,252 |
| Ref-SAV | 72,509 |
| Total | 459,052 |

All seven training sources use their configured training expressions, without the former private holdout. ReVOS includes 114 expressions with explicit `anno_id: []`; missing/null annotation IDs still raise errors. Ref-SAV keeps the full training expression list. The index builder checks Ref-SAV validation video-ID separation from training; this is not content-level duplicate detection.

`focusseg_r1` supplies shared index/decoding/sampling utilities. Its presence is a code dependency, not evidence that the current model uses the historical query architecture.

The recipe differs from the original Sa2VA multitask training: it excludes QA/GCG/VP mixtures, uses expression-based sampling, uses Video-R1 as the language backbone, and freezes SAM2. These differences should accompany reported comparisons.

## Optimization, validation, and results

- Four A800 GPUs; microbatch one per GPU; accumulation eight; effective batch 32.
- Three epochs; 14,346 optimizer steps per epoch; 43,038 total steps. Epoch-tail padding can repeat a small number of expressions.
- AdamW, betas (0.9, 0.999), weight decay 0.05, gradient clipping 1.0.
- Learning rates: 4e-5 for LoRA/task-token rows and 4e-6 for the projection; 5% warmup followed by cosine decay across the original total budget.
- Full Ref-SAV validation every 2,000 steps and at epoch boundaries, with 1,945 expressions and full videos.
- J is region IoU; F uses the repository's DAVIS-style boundary scorer with threshold 0.008. `Overall` averages expressions. `PaperBalanced` averages the Long and Short group means equally, separately for J, F, and J&F. The latter selects the best checkpoint.

This is evaluation on the official validation split using the local scorer, not a claim that the scorer is the official released implementation. Since validation selects checkpoints, it is not an untouched test set.

The user-reported step-14,346 checkpoint achieved Overall J&F 63.7710 and PaperBalanced J&F 58.0556. The locally evaluated baseline achieved 63.8044 and 57.4708 respectively. These results do not show an improvement in both aggregations and do not establish statistical significance. The repository does not bundle the corresponding weights or raw evaluation artifacts.

## Running on the original cluster

The launchers contain the original cluster's project, Python, partition, and dataset paths. For another environment, edit them and the JSON before preparing a new index and smoke report. Keep the configured models and required dataset files available separately. The original environment used Python 3.10, torch 2.5.1+cu124, peft 0.19.1, and transformers 4.57.6, plus the package's existing SAM2 and video/image decoding dependencies.

Run from the repository root with the intended environment active:

```bash
export PYTHONPATH="$PWD:$PWD/qwen_vl_utils/src${PYTHONPATH:+:$PYTHONPATH}"
python -m unittest discover -s tests -p 'test_seg_lora*.py' -v
```

For the original Slurm-compatible cluster, first prepare the index:

```bash
mkdir -p logs
yhbatch --gres=gpu:a800:1 \
  --output='logs/%j.seg_lora_index.out' \
  --error='logs/%j.seg_lora_index.err' \
  scripts/yhbatch_seg_lora_index.sh
```

Use the successful preparation job's SQLite path and its `.ready.json` sidecar. Then submit smoke and, after it succeeds, fresh training:

```bash
# Replace the example paths with paths from your successful jobs.
INDEX=/absolute/path/to/seg_lora_full_INDEX_JOB.sqlite
yhbatch --gres=gpu:a800:4 \
  --output='logs/%j.seg_lora_smoke.out' \
  --error='logs/%j.seg_lora_smoke.err' \
  scripts/yhbatch_seg_lora_4gpu.sh smoke "$INDEX"

SMOKE=/absolute/path/to/seg_lora_smoke_SMOKE_JOB/smoke.json
yhbatch --gres=gpu:a800:4 \
  --output='logs/%j.seg_lora_train.out' \
  --error='logs/%j.seg_lora_train.err' \
  scripts/yhbatch_seg_lora_4gpu.sh train "$INDEX" "$SMOKE"
```

Fresh training reloads original pretrained weights; it does not train from the smoke-test updates. Code/configuration/index provenance must match the successful smoke report.

## Checkpoints and resuming

The checkpoint contains trainable parameters, optimizer and scheduler state, four-rank RNG state, current step, and provenance. It is not a standalone Hugging Face full-model export: loading also requires the original pretrained models and this implementation.

`seg_lora_storage.py` writes a temporary file in the output directory, flushes it, then atomically replaces the checkpoint. Only after a successful save does retention remove obsolete numbered checkpoints, keeping latest and best. Validation reports remain. Saving requires temporary headroom for a third checkpoint. `/var/tmp` is used for job-local temporary/cache files, not for persistent checkpoint storage.

The original interrupted run resumes in `outputs/seg_lora_train_99970`. Read `latest.json`, verify its recorded SHA256, and pass that file explicitly:

```bash
CKPT=/absolute/path/to/seg_lora_train_99970/checkpoint_00016000.pt
yhbatch --gres=gpu:a800:4 \
  --output='logs/%j.seg_lora_resume.out' \
  --error='logs/%j.seg_lora_resume.err' \
  scripts/yhbatch_seg_lora_resume_4gpu.sh "$CKPT"
```

That launcher also requires the original run's index, smoke report, and `backups/seg_lora_storage_20261008/original_run.json`. These are runtime artifacts and are intentionally not uploaded. `--storage-upgrade-reference` allows the reviewed storage-only migration while continuing to reject model, configuration, or data identity changes. Do not use the historical migration snapshot for unrelated fresh runs. The total budget remains 43,038 updates, rather than adding three new epochs.

## Upstream assets

This work builds on Sa2VA, Video-R1, and SAM2. Follow the applicable upstream code, model, and dataset terms. This snapshot adds no new blanket license for third-party components.
