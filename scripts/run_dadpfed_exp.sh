#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Editable experiment settings
# -----------------------------
DATASET="${DATASET:-UCF101-MA}"          # UCF101-MA | CIFAR10 | CIFAR100 | face_dataset | hmdb51
METHOD="${METHOD:-DADPFed}"              # DADPFed | DADPFedSAM
NON_IID="${NON_IID:-1}"                  # 1: enable Dirichlet non-IID, 0: IID
SPLIT_RULE="${SPLIT_RULE:-Dirichlet}"    # Dirichlet | Pathological
SPLIT_COEF="${SPLIT_COEF:-0.3}"          # D1=0.3, D2=0.6, IID-like=100
TOTAL_CLIENT="${TOTAL_CLIENT:-100}"
ACTIVE_RATIO="${ACTIVE_RATIO:-0.15}"     # 1.0 for full participation; 0.15 for partial participation
COMM_ROUNDS="${COMM_ROUNDS:-1000}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
SEED="${SEED:-20}"
CUDA_ID="${CUDA_ID:-0}"
DATA_FILE="${DATA_FILE:-./}"
OUT_FILE="${OUT_FILE:-out/}"

# DADPFed hyperparameters
GLOBAL_LR="${GLOBAL_LR:-1.0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.001}"
ALPHA="${ALPHA:-0.1}"
RHO="${RHO:-0.05}"                       # only used by DADPFedSAM
DADPFED_CYCLE="${DADPFED_CYCLE:-40}"     # R
DADPFED_RETENTION="${DADPFED_RETENTION:-0.75}"  # Th
DADPFED_MASK_QUANTILE="${DADPFED_MASK_QUANTILE:-0.75}"

# Dataset-specific defaults (can still be overridden by env vars)
if [[ "$DATASET" == "UCF101-MA" ]]; then
  MODEL="${MODEL:-LeNet}"
  BATCH_SIZE="${BATCH_SIZE:-8}"
  LOCAL_LR="${LOCAL_LR:-0.01}"
elif [[ "$DATASET" == "CIFAR10" || "$DATASET" == "CIFAR100" ]]; then
  MODEL="${MODEL:-ResNet18}"
  BATCH_SIZE="${BATCH_SIZE:-64}"
  LOCAL_LR="${LOCAL_LR:-0.01}"
else
  MODEL="${MODEL:-ResNet18}"
  BATCH_SIZE="${BATCH_SIZE:-16}"
  LOCAL_LR="${LOCAL_LR:-0.01}"
fi

CMD=(
  python -u train.py
  --dataset "$DATASET"
  --model "$MODEL"
  --method "$METHOD"
  --split-rule "$SPLIT_RULE"
  --split-coef "$SPLIT_COEF"
  --total-client "$TOTAL_CLIENT"
  --active-ratio "$ACTIVE_RATIO"
  --comm-rounds "$COMM_ROUNDS"
  --local-epochs "$LOCAL_EPOCHS"
  --batchsize "$BATCH_SIZE"
  --local-learning-rate "$LOCAL_LR"
  --global-learning-rate "$GLOBAL_LR"
  --weight-decay "$WEIGHT_DECAY"
  --alpha "$ALPHA"
  --rho "$RHO"
  --dadpfed-cycle "$DADPFED_CYCLE"
  --dadpfed-retention "$DADPFED_RETENTION"
  --dadpfed-mask-quantile "$DADPFED_MASK_QUANTILE"
  --seed "$SEED"
  --cuda "$CUDA_ID"
  --data-file "$DATA_FILE"
  --out-file "$OUT_FILE"
)

if [[ "$NON_IID" == "1" ]]; then
  CMD+=(--non-iid)
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
