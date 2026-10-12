#!/usr/bin/env bash
set -eo pipefail
PROJ=/XYFS02/HDD_POOL/lzu_bmwang/lzu_bmwang_1/codefile/zhangyifan/Sa2VA_Pro
PY=/HOME/lzu_bmwang/lzu_bmwang_1/.conda/envs/sa2va_pro/bin/python
cd "$PROJ"
export PYTHONPATH="$PROJ:$PROJ/qwen_vl_utils/src${PYTHONPATH:+:$PYTHONPATH}"
"$PY" -m compileall -q seg_lora local_scripts/seg_lora_run.py
"$PY" -c 'import torch, peft, transformers; from seg_lora.model import SegLoRA; print("SEG_LORA_IMPORT_OK", torch.__version__, peft.__version__, transformers.__version__)'
"$PY" -m unittest discover -s tests -p 'test_seg_lora*.py' -v
bash -n scripts/yhbatch_seg_lora_index.sh
bash -n scripts/yhbatch_seg_lora_4gpu.sh
echo SEG_LORA_CPU_PREFLIGHT_OK
