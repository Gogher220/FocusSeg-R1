#!/usr/bin/env bash
set -euo pipefail

if [[ "${ALLOW_SFT:-0}" != "1" ]]; then
  echo "SFT is disabled for the current Sa2VA_Pro experiment plan. Set ALLOW_SFT=1 only if you explicitly want to run SFT." >&2
  exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src:${PROJECT_ROOT}/r1_v/src:${PROJECT_ROOT}/qwen_vl_utils/src:${PYTHONPATH:-}"

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
DATASET_JSON="${DATASET_JSON:-${DATA_ROOT}/train_sft.json}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-VL-2B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/sa2va_pro_sft}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

python -m torch.distributed.run --nproc_per_node="${NPROC_PER_NODE}" \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port=12349 \
  src/open_r1/sft_video.py \
  --output_dir "${OUTPUT_DIR}" \
  --model_name_or_path "${MODEL_NAME}" \
  --dataset_name "${DATASET_JSON}" \
  --data_root "${DATA_ROOT}" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1e-6 \
  --logging_steps 1 \
  --bf16 true \
  --gradient_checkpointing true \
  --num_train_epochs 1 \
  --save_steps 500 \
  --max_grad_norm 5
