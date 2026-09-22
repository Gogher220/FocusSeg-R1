#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src:${PROJECT_ROOT}/r1_v/src:${PROJECT_ROOT}/qwen_vl_utils/src:${PYTHONPATH:-}"

DATA_ROOT="${DATA_ROOT:-/XYAIFS00/HDD_POOL/nsccgz_zgchen/nsccgz_zgchen_6/data/sa2va_pro}"
PROC_ROOT="${PROC_ROOT:-${DATA_ROOT}/processed}"
MODEL_NAME="${MODEL_NAME:-/XYAIFS00/HDD_POOL/nsccgz_zgchen/nsccgz_zgchen_6/pretrained/Qwen2.5-VL-2B-Instruct}"
SFT_DATASET_JSON="${SFT_DATASET_JSON:-${PROC_ROOT}/train_sft_all.jsonl}"
SFT_OUTPUT_DIR="${SFT_OUTPUT_DIR:-${PROJECT_ROOT}/outputs/sa2va_pro_sft_all}"
GRPO_DATASET_JSON="${GRPO_DATASET_JSON:-${PROC_ROOT}/train_sft_all.jsonl}"
GRPO_OUTPUT_DIR="${GRPO_OUTPUT_DIR:-${PROJECT_ROOT}/outputs/sa2va_pro_grpo_all}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
RUN_GRPO="${RUN_GRPO:-0}"

export DATA_ROOT
export CUDA_VISIBLE_DEVICES
export NPROC_PER_NODE

cat >&2 <<'EOF'
This legacy pipeline runs SFT before GRPO and is disabled for the current
Sa2VA_Pro experiment plan. Use scripts/run_grpo_sa2va_pro.sh or the A800
yhbatch_grpo_*.sh scripts to run S-GRPO from the original Video-R1 checkpoint.
EOF
exit 1
