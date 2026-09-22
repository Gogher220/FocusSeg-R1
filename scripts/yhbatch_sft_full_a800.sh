#!/bin/bash
#SBATCH --job-name=sa2va_sft
#SBATCH -p ip001_bmw
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -o /XYFS02/HDD_POOL/lzu_bmwang/lzu_bmwang_1/codefile/zhangyifan/Sa2VA_Pro/logs/%j.out
#SBATCH -e /XYFS02/HDD_POOL/lzu_bmwang/lzu_bmwang_1/codefile/zhangyifan/Sa2VA_Pro/logs/%j.err

set -euo pipefail

if [[ "${ALLOW_SFT:-0}" != "1" ]]; then
  echo "SFT is disabled for the current Sa2VA_Pro experiment plan. Set ALLOW_SFT=1 only if you explicitly want to run SFT." >&2
  exit 1
fi

PROJ="${PROJ:-/XYFS02/HDD_POOL/lzu_bmwang/lzu_bmwang_1/codefile/zhangyifan/Sa2VA_Pro}"
ENV_PATH="${ENV_PATH:-$HOME/.conda/envs/sa2va_pro}"
DATA_ROOT="${DATA_ROOT:-/XYFS02/HDD_POOL/lzu_bmwang/lzu_bmwang_1/data/Sa2VA-data}"
TRAIN_JSON="${TRAIN_JSON:-/XYFS02/HDD_POOL/lzu_bmwang/lzu_bmwang_1/data/sa2va_pro/processed/davis17_train.jsonl}"
MODEL_NAME="${MODEL_NAME:-/XYFS02/HDD_POOL/lzu_bmwang/lzu_bmwang_1/pretrained/Video-R1Qwen2.5-VL-7B-COT-SFT}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJ/outputs/sa2va_pro_sft_davis17_full}"
RESUME_CKPT="${RESUME_CKPT:-}"
MASTER_PORT="${MASTER_PORT:-12350}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
SAM2_CKPT="${SAM2_CKPT:-/XYFS02/HDD_POOL/lzu_bmwang/lzu_bmwang_1/pretrained/Sa2VA/sam2_hiera_large.pt}"
SEG_LOSS_WEIGHT="${SEG_LOSS_WEIGHT:-0.001}"
OPTIM="${OPTIM:-adamw_torch_fused}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
USE_PEFT="${USE_PEFT:-false}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
FREEZE_VIDEO_R1="${FREEZE_VIDEO_R1:-false}"
TRAINABLE_PARAM_PATTERNS="${TRAINABLE_PARAM_PATTERNS:-text_hidden_fcs,lora_,modules_to_save}"

mkdir -p "$PROJ/logs" "$PROJ/outputs"
RUNTIME_CACHE="${RUNTIME_CACHE:-/XYFS02/HDD_POOL/lzu_bmwang/lzu_bmwang_1/runtime_cache/sa2va_pro}"
mkdir -p \
  "$RUNTIME_CACHE" \
  "$RUNTIME_CACHE/xdg" \
  "$RUNTIME_CACHE/hf" \
  "$RUNTIME_CACHE/hf/datasets" \
  "$RUNTIME_CACHE/hf/hub" \
  "$RUNTIME_CACHE/hf/transformers" \
  "$RUNTIME_CACHE/tmp" \
  "$RUNTIME_CACHE/triton" \
  "$RUNTIME_CACHE/torch_extensions"
export XDG_CACHE_HOME="$RUNTIME_CACHE/xdg"
export HF_HOME="$RUNTIME_CACHE/hf"
export HF_DATASETS_CACHE="$RUNTIME_CACHE/hf/datasets"
export HF_HUB_CACHE="$RUNTIME_CACHE/hf/hub"
export TRANSFORMERS_CACHE="$RUNTIME_CACHE/hf/transformers"
export TMPDIR="$RUNTIME_CACHE/tmp"
export TEMP="$RUNTIME_CACHE/tmp"
export TMP="$RUNTIME_CACHE/tmp"
export TRITON_CACHE_DIR="$RUNTIME_CACHE/triton"
export TORCH_EXTENSIONS_DIR="$RUNTIME_CACHE/torch_extensions"
export SA2VA_PRO_SAM2_CKPT="$SAM2_CKPT"
export SA2VA_PRO_SEG_LOSS_WEIGHT="$SEG_LOSS_WEIGHT"
export PYTORCH_CUDA_ALLOC_CONF

source /APP/u22/ai_x86/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate "$ENV_PATH"

if [[ -z "${CUDA_HOME:-}" ]]; then
  for candidate in /usr/local/cuda-12.4 /usr/local/cuda-12.2 /usr/local/cuda /APP/u22/ai_x86/cuda-12.4 /APP/u22/ai_x86/cuda-12.2 /APP/u22/ai_x86/cuda; do
    if [[ -x "$candidate/bin/nvcc" ]]; then
      export CUDA_HOME="$candidate"
      break
    fi
  done
fi

if [[ -n "${CUDA_HOME:-}" ]]; then
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi

unset PYTHONPATH PYTHONHOME PYTHONUSERBASE PIP_USER
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJ:$PROJ/src:$PROJ/r1_v/src:$PROJ/qwen_vl_utils/src"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd "$PROJ"

EXTRA_ARGS=()
if [[ -n "$RESUME_CKPT" ]]; then
  EXTRA_ARGS+=(--resume_from_checkpoint "$RESUME_CKPT")
fi

PEFT_ARGS=()
if [[ "$USE_PEFT" == "true" ]]; then
  PEFT_ARGS+=(--use_peft true)
  PEFT_ARGS+=(--lora_r "$LORA_R")
  PEFT_ARGS+=(--lora_alpha "$LORA_ALPHA")
  PEFT_ARGS+=(--lora_dropout "$LORA_DROPOUT")
fi

FREEZE_ARGS=()
if [[ "$FREEZE_VIDEO_R1" == "true" ]]; then
  FREEZE_ARGS+=(--freeze_video_r1 true)
  FREEZE_ARGS+=(--trainable_param_patterns "$TRAINABLE_PARAM_PATTERNS")
fi

echo "===== JOB INFO ====="
echo "HOSTNAME=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CUDA_HOME=${CUDA_HOME:-}"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "TRAIN_JSON=$TRAIN_JSON"
echo "SAM2_CKPT=$SAM2_CKPT"
echo "SEG_LOSS_WEIGHT=$SEG_LOSS_WEIGHT"
echo "OPTIM=$OPTIM"
echo "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo "USE_PEFT=$USE_PEFT"
echo "FREEZE_VIDEO_R1=$FREEZE_VIDEO_R1"
echo "TRAINABLE_PARAM_PATTERNS=$TRAINABLE_PARAM_PATTERNS"
if [[ "$USE_PEFT" == "true" ]]; then
  echo "LORA_R=$LORA_R"
  echo "LORA_ALPHA=$LORA_ALPHA"
  echo "LORA_DROPOUT=$LORA_DROPOUT"
fi
if [[ -n "$RESUME_CKPT" ]]; then
  echo "RESUME_CKPT=$RESUME_CKPT"
fi

echo "===== GPU CHECK ====="
nvidia-smi
which nvcc || true
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"device{i}:", torch.cuda.get_device_name(i))
PY

yhrun python -m torch.distributed.run --nproc_per_node="$NPROC_PER_NODE" \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port="$MASTER_PORT" \
  src/open_r1/sft_video.py \
  --output_dir "$OUTPUT_DIR" \
  --model_name_or_path "$MODEL_NAME" \
  --dataset_name "$TRAIN_JSON" \
  --data_root "$DATA_ROOT" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-6 \
  --optim "$OPTIM" \
  --logging_steps 10 \
  --bf16 true \
  --gradient_checkpointing true \
  --num_train_epochs 1 \
  --save_steps 100 \
  --save_total_limit 3 \
  --max_grad_norm 5 \
  "${PEFT_ARGS[@]}" \
  "${FREEZE_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
