#!/usr/bin/env bash
#SBATCH --job-name=seg_lora_index
#SBATCH --partition=ip001_bmw
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=8
set -eo pipefail
source /APP/u22/ai_x86/toolshs/set-XY-I.sh
set -u
PROJ=/XYFS02/HDD_POOL/lzu_bmwang/lzu_bmwang_1/codefile/zhangyifan/Sa2VA_Pro
PY=/HOME/lzu_bmwang/lzu_bmwang_1/.conda/envs/sa2va_pro/bin/python
cd "$PROJ"
export PYTHONPATH="$PROJ:$PROJ/qwen_vl_utils/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2
"$PY" -u -m local_scripts.seg_lora_run --mode prepare \
  --config configs/seg_lora_fresh_3epoch.json \
  --index "data_indexes/seg_lora_full_${SLURM_JOB_ID}.sqlite" \
  --out "outputs/seg_lora_index_${SLURM_JOB_ID}"
echo "INDEX_JOB_FINISHED INDEX=$PROJ/data_indexes/seg_lora_full_${SLURM_JOB_ID}.sqlite"
