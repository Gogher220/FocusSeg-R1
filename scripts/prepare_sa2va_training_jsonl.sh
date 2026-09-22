#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src:${PROJECT_ROOT}/r1_v/src:${PROJECT_ROOT}/qwen_vl_utils/src:${PYTHONPATH:-}"

TRAIN_ROOT="${TRAIN_ROOT:-/XYAIFS00/HDD_POOL/nsccgz_zgchen/nsccgz_zgchen_6/data/Sa2VA-Training}"
OUT_DIR="${OUT_DIR:-/XYAIFS00/HDD_POOL/nsccgz_zgchen/nsccgz_zgchen_6/data/sa2va_pro/processed}"
MAX_PER_FILE="${MAX_PER_FILE:-0}"
DATA_TYPES="${DATA_TYPES:-image,video}"

python local_scripts/prepare_sa2va_training_jsonl.py \
  --train_root "${TRAIN_ROOT}" \
  --out_dir "${OUT_DIR}" \
  --max_per_file "${MAX_PER_FILE}" \
  --data_types "${DATA_TYPES}"
