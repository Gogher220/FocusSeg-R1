#!/usr/bin/env bash
#SBATCH --job-name=seg_lora_fresh
#SBATCH --partition=ip001_bmw
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a800:4
#SBATCH --cpus-per-task=16
set -eo pipefail
source /APP/u22/ai_x86/toolshs/set-XY-I.sh
set -u
PROJ=/XYFS02/HDD_POOL/lzu_bmwang/lzu_bmwang_1/codefile/zhangyifan/Sa2VA_Pro
PY=/HOME/lzu_bmwang/lzu_bmwang_1/.conda/envs/sa2va_pro/bin/python
MODE="${1:?smoke or train}"
INDEX="${2:?New complete index absolute path}"
case "$MODE" in smoke|train) ;; *) exit 2;; esac
test -f "$INDEX"
EXTRA=()
if [ "$MODE" = train ]; then
  REPORT="${3:?Successful smoke.json path, or explicit --skip-smoke}"
  if [ "$REPORT" = --skip-smoke ]; then
    EXTRA+=(--skip-smoke)
  else
    test -f "$REPORT"
    EXTRA+=(--smoke-report "$REPORT")
  fi
fi
cd "$PROJ"
export PYTHONPATH="$PROJ:$PROJ/qwen_vl_utils/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2
OUT="$PROJ/outputs/seg_lora_${MODE}_${SLURM_JOB_ID}"
echo "NODE=$(hostname) JOB_ID=$SLURM_JOB_ID MODE=$MODE INDEX=$INDEX OUT=$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -u -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 \
  -m local_scripts.seg_lora_run --mode "$MODE" --config configs/seg_lora_fresh_3epoch.json \
  --index "$INDEX" --out "$OUT" "${EXTRA[@]}"
echo "SEG_LORA_JOB_FINISHED OUT=$OUT"
