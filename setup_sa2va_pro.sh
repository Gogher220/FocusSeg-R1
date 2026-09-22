#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-sa2va_pro}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[1/5] create or update conda env: ${ENV_NAME}"
conda env update -n "${ENV_NAME}" -f "${PROJECT_ROOT}/sa2va_pro.env.yml" --prune || \
conda env create -n "${ENV_NAME}" -f "${PROJECT_ROOT}/sa2va_pro.env.yml"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "[2/5] install qwen_vl_utils editable"
cd "${PROJECT_ROOT}/qwen_vl_utils"
pip install -e .[decord]

echo "[3/5] install training package editable"
cd "${PROJECT_ROOT}/r1_v"
pip install -e .

echo "[4/5] export python path hints"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src:${PROJECT_ROOT}/r1_v/src:${PROJECT_ROOT}/qwen_vl_utils/src:${PYTHONPATH:-}"

echo "[5/5] smoke check"
python - <<'PY'
from configuration_sa2va_pro import Sa2VAProConfig
from sa2va_pro_utils import SEG_TOKEN
print("config ok", Sa2VAProConfig.model_type, SEG_TOKEN)
PY

echo "Done. Activate with: conda activate ${ENV_NAME}"
