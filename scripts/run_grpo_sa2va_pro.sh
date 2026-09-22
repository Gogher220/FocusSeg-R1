#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src:${PROJECT_ROOT}/r1_v/src:${PROJECT_ROOT}/qwen_vl_utils/src:${PYTHONPATH:-}"
export DEBUG_MODE="${DEBUG_MODE:-true}"
export LOG_PATH="${LOG_PATH:-${PROJECT_ROOT}/grpo_debug.txt}"

if [[ -z "${CUDA_HOME:-}" ]]; then
  if command -v nvcc >/dev/null 2>&1; then
    export CUDA_HOME
    CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
  elif [[ -d "/usr/local/cuda" ]]; then
    export CUDA_HOME="/usr/local/cuda"
  fi
fi

if [[ -n "${CUDA_HOME:-}" ]]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi

DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data}"
DATASET_JSON="${DATASET_JSON:-${DATA_ROOT}/train_grpo.json}"
MODEL_NAME="${MODEL_NAME:-/XYFS02/HDD_POOL/lzu_bmwang/lzu_bmwang_1/pretrained/Video-R1Qwen2.5-VL-7B-COT-SFT}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/sa2va_pro_sgrpo}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
USE_PEFT="${USE_PEFT:-true}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
FREEZE_VIDEO_R1="${FREEZE_VIDEO_R1:-true}"
TRAINABLE_PARAM_PATTERNS="${TRAINABLE_PARAM_PATTERNS:-lora_}"

PEFT_ARGS=()
if [[ "${USE_PEFT}" == "true" ]]; then
  PEFT_ARGS+=(--use_peft true)
  PEFT_ARGS+=(--lora_r "${LORA_R}")
  PEFT_ARGS+=(--lora_alpha "${LORA_ALPHA}")
  PEFT_ARGS+=(--lora_dropout "${LORA_DROPOUT}")
fi

FREEZE_ARGS=()
if [[ "${FREEZE_VIDEO_R1}" == "true" ]]; then
  FREEZE_ARGS+=(--freeze_video_r1 true)
  FREEZE_ARGS+=(--trainable_param_patterns "${TRAINABLE_PARAM_PATTERNS}")
fi

python -m torch.distributed.run --nproc_per_node="${NPROC_PER_NODE}" \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port=12365 \
  src/open_r1/grpo.py \
  --output_dir "${OUTPUT_DIR}" \
  --model_name_or_path "${MODEL_NAME}" \
  --dataset_name "${DATASET_JSON}" \
  --data_root "${DATA_ROOT}" \
  --max_prompt_length 16384 \
  --max_completion_length 768 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-6 \
  --lr_scheduler_type cosine \
  --weight_decay 0.01 \
  --bf16 true \
  --logging_steps 1 \
  --gradient_checkpointing true \
  --temporal true \
  --temporal_reward_weight 0.35 \
  --temporal_threshold_ratio 0.85 \
  --spatial true \
  --spatial_reward_weight 0.35 \
  --spatial_keep_ratio 0.65 \
  --spatial_occ_ratio_min 0.10 \
  --spatial_occ_ratio_max 0.30 \
  --spatial_occ_frame_ratio 0.4 \
  --spatial_threshold_ratio 0.85 \
  --len_control true \
  --max_pixels 401408 \
  --num_train_epochs 1 \
  --save_steps 100 \
  --beta 0.03 \
  --max_grad_norm 5 \
  --num_generations 4 \
  --reward_funcs accuracy segmentation_iou segmentation_presence \
  "${PEFT_ARGS[@]}" \
  "${FREEZE_ARGS[@]}"
