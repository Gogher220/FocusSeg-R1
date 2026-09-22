#!/bin/bash
#SBATCH --job-name=sa2va_grpo
#SBATCH -p ip001_bmw
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -o /XYFS02/HDD_POOL/lzu_bmwang/lzu_bmwang_1/codefile/zhangyifan/Sa2VA_Pro/logs/%j.out
#SBATCH -e /XYFS02/HDD_POOL/lzu_bmwang/lzu_bmwang_1/codefile/zhangyifan/Sa2VA_Pro/logs/%j.err

set -euo pipefail

PROJ="${PROJ:-/XYFS02/HDD_POOL/lzu_bmwang/lzu_bmwang_1/codefile/zhangyifan/Sa2VA_Pro}"
ENV_PATH="${ENV_PATH:-$HOME/.conda/envs/sa2va_pro}"
DATA_ROOT="${DATA_ROOT:-/XYFS02/HDD_POOL/lzu_bmwang/lzu_bmwang_1/data/Sa2VA-data}"
TRAIN_JSON="${TRAIN_JSON:-/XYFS02/HDD_POOL/lzu_bmwang/lzu_bmwang_1/data/sa2va_pro/processed/davis17_train.jsonl}"
MODEL_NAME="${MODEL_NAME:-/XYFS02/HDD_POOL/lzu_bmwang/lzu_bmwang_1/pretrained/Video-R1Qwen2.5-VL-7B-COT-SFT}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJ/outputs/sa2va_pro_sgrpo_davis17}"
RESUME_CKPT="${RESUME_CKPT:-}"
MASTER_PORT="${MASTER_PORT:-12365}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
SAM2_CKPT="${SAM2_CKPT:-/XYFS02/HDD_POOL/lzu_bmwang/lzu_bmwang_1/pretrained/Sa2VA/sam2_hiera_large.pt}"
SEG_LOSS_WEIGHT="${SEG_LOSS_WEIGHT:-0.001}"
TEMPORAL="${TEMPORAL:-true}"
SPATIAL="${SPATIAL:-true}"
LEN_CONTROL="${LEN_CONTROL:-true}"
TEMPORAL_REWARD_WEIGHT="${TEMPORAL_REWARD_WEIGHT:-0.35}"
TEMPORAL_THRESHOLD_RATIO="${TEMPORAL_THRESHOLD_RATIO:-0.85}"
SPATIAL_REWARD_WEIGHT="${SPATIAL_REWARD_WEIGHT:-0.35}"
SPATIAL_KEEP_RATIO="${SPATIAL_KEEP_RATIO:-0.65}"
SPATIAL_OCC_RATIO_MIN="${SPATIAL_OCC_RATIO_MIN:-0.10}"
SPATIAL_OCC_RATIO_MAX="${SPATIAL_OCC_RATIO_MAX:-0.30}"
SPATIAL_OCC_FRAME_RATIO="${SPATIAL_OCC_FRAME_RATIO:-0.4}"
SPATIAL_THRESHOLD_RATIO="${SPATIAL_THRESHOLD_RATIO:-0.85}"
REWARD_FUNCS="${REWARD_FUNCS:-accuracy segmentation_iou segmentation_presence}"
USE_PEFT="${USE_PEFT:-true}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
FREEZE_VIDEO_R1="${FREEZE_VIDEO_R1:-true}"
TRAINABLE_PARAM_PATTERNS="${TRAINABLE_PARAM_PATTERNS:-lora_}"
MAX_STEPS="${MAX_STEPS:-}"

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

source /APP/u22/ai_x86/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate "$ENV_PATH"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PATH/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

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
export DEBUG_MODE="${DEBUG_MODE:-false}"
export LOG_PATH="${LOG_PATH:-$OUTPUT_DIR/grpo_debug.log}"

cd "$PROJ"

EXTRA_ARGS=()
if [[ -n "$RESUME_CKPT" ]]; then
  EXTRA_ARGS+=(--resume_from_checkpoint "$RESUME_CKPT")
fi

read -r -a REWARD_FUNCS_ARRAY <<< "$REWARD_FUNCS"

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

LIMIT_ARGS=()
if [[ -n "$MAX_STEPS" ]]; then
  LIMIT_ARGS+=(--max_steps "$MAX_STEPS")
fi

echo "===== JOB INFO ====="
echo "HOSTNAME=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CUDA_HOME=${CUDA_HOME:-}"
echo "MODEL_NAME=$MODEL_NAME"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "TRAIN_JSON=$TRAIN_JSON"
echo "SAM2_CKPT=$SAM2_CKPT"
echo "SEG_LOSS_WEIGHT=$SEG_LOSS_WEIGHT"
echo "TEMPORAL=$TEMPORAL"
echo "SPATIAL=$SPATIAL"
echo "LEN_CONTROL=$LEN_CONTROL"
echo "REWARD_FUNCS=$REWARD_FUNCS"
echo "USE_PEFT=$USE_PEFT"
echo "FREEZE_VIDEO_R1=$FREEZE_VIDEO_R1"
echo "TRAINABLE_PARAM_PATTERNS=$TRAINABLE_PARAM_PATTERNS"
echo "NPROC_PER_NODE=$NPROC_PER_NODE"
if [[ "$USE_PEFT" == "true" ]]; then
  echo "LORA_R=$LORA_R"
  echo "LORA_ALPHA=$LORA_ALPHA"
  echo "LORA_DROPOUT=$LORA_DROPOUT"
fi
if [[ -n "$MAX_STEPS" ]]; then
  echo "MAX_STEPS=$MAX_STEPS"
fi
if [[ -n "$RESUME_CKPT" ]]; then
  echo "RESUME_CKPT=$RESUME_CKPT"
fi

echo "===== GPU CHECK ====="
nvidia-smi
which nvcc || true
"$PYTHON_BIN" - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"device{i}:", torch.cuda.get_device_name(i))
PY

yhrun "$PYTHON_BIN" -m torch.distributed.run --nproc_per_node="$NPROC_PER_NODE" \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port="$MASTER_PORT" \
  src/open_r1/grpo.py \
  --output_dir "$OUTPUT_DIR" \
  --model_name_or_path "$MODEL_NAME" \
  --dataset_name "$TRAIN_JSON" \
  --data_root "$DATA_ROOT" \
  --max_prompt_length 16384 \
  --max_completion_length 768 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --learning_rate 5e-7 \
  --lr_scheduler_type cosine \
  --weight_decay 0.01 \
  --bf16 true \
  --logging_steps 5 \
  --gradient_checkpointing true \
  --temporal "$TEMPORAL" \
  --temporal_reward_weight "$TEMPORAL_REWARD_WEIGHT" \
  --temporal_threshold_ratio "$TEMPORAL_THRESHOLD_RATIO" \
  --spatial "$SPATIAL" \
  --spatial_reward_weight "$SPATIAL_REWARD_WEIGHT" \
  --spatial_keep_ratio "$SPATIAL_KEEP_RATIO" \
  --spatial_occ_ratio_min "$SPATIAL_OCC_RATIO_MIN" \
  --spatial_occ_ratio_max "$SPATIAL_OCC_RATIO_MAX" \
  --spatial_occ_frame_ratio "$SPATIAL_OCC_FRAME_RATIO" \
  --spatial_threshold_ratio "$SPATIAL_THRESHOLD_RATIO" \
  --len_control "$LEN_CONTROL" \
  --max_pixels 401408 \
  --num_train_epochs 1 \
  --save_steps 50 \
  --save_total_limit 3 \
  --beta 0.03 \
  --max_grad_norm 5 \
  --num_generations 4 \
  --reward_funcs "${REWARD_FUNCS_ARRAY[@]}" \
  "${PEFT_ARGS[@]}" \
  "${LIMIT_ARGS[@]}" \
  "${FREEZE_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
