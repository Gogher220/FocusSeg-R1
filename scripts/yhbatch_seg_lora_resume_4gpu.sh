#!/usr/bin/env bash
#SBATCH --job-name=seg_lora_resume
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
RUN="$PROJ/outputs/seg_lora_train_99970"
CKPT="${1:-$RUN/checkpoint_00016000.pt}"
REFERENCE="$PROJ/backups/seg_lora_storage_20261008/original_run.json"
test -s "$CKPT"
test -s "$REFERENCE"
cd "$PROJ"
export PYTHONPATH="$PROJ:$PROJ/qwen_vl_utils/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2
umask 077
export TMPDIR
TMPDIR=$(mktemp -d "/var/tmp/seg_lora_${SLURM_JOB_ID}.XXXXXX")
export TORCH_EXTENSIONS_DIR="$TMPDIR/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/torch_inductor"
echo "COMPUTE_NODE_TEMP=$TMPDIR"
echo "NODE=$(hostname) JOB_ID=$SLURM_JOB_ID RESUME=$CKPT POLICY=latest_and_best OUT=$RUN"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -u -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 \
  -m local_scripts.seg_lora_run --mode train --config configs/seg_lora_fresh_3epoch.json \
  --index "$PROJ/data_indexes/seg_lora_full_99644.sqlite" --out "$RUN" \
  --smoke-report "$PROJ/outputs/seg_lora_smoke_99799/smoke.json" \
  --resume "$CKPT" --storage-upgrade-reference "$REFERENCE"
echo "SEG_LORA_RESUME_JOB_FINISHED OUT=$RUN"
