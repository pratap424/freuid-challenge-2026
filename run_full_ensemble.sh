#!/bin/bash
# =============================================================
# FREUID Challenge 2026 - Full Ensemble Training
# Run AFTER baseline is confirmed working
# This trains ALL models across ALL folds
# =============================================================

set -e

CONDA_ENV="vlm_distill"
BASE_DIR="$HOME/freuid_challenge"
DATA_DIR="$BASE_DIR/data"

source activate $CONDA_ENV 2>/dev/null || conda activate $CONDA_ENV
cd $BASE_DIR

echo "=============================================="
echo "  FULL ENSEMBLE TRAINING"  
echo "  $(date)"
echo "=============================================="

# ===========================
# Train all models, all folds
# ===========================
MODELS=("eva02" "convnextv2" "swin" "effnetv2" "vit_clip")
# Note: dinov2 needs more VRAM, train separately if needed

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo ">>> Training $MODEL - All 5 folds"
    python train.py \
        --model $MODEL \
        --fold all \
        --epochs 25 \
        --aug_level heavy \
        --use_ela \
        --data_dir $DATA_DIR \
        2>&1 | tee logs/train_${MODEL}.log
    
    echo ""
    echo ">>> Inference $MODEL with TTA"
    python inference.py \
        --model $MODEL \
        --fold all \
        --tta \
        --data_dir $DATA_DIR \
        2>&1 | tee logs/inference_${MODEL}.log
done

# ===========================
# Ensemble
# ===========================
echo ""
echo ">>> Running ensemble optimization..."
python ensemble.py \
    --oof_dir $BASE_DIR/checkpoints \
    --pred_dir $BASE_DIR/outputs/predictions \
    --train_csv $DATA_DIR/train.csv \
    --output_dir $BASE_DIR/outputs/submissions

echo ""
echo "=============================================="
echo "  ENSEMBLE TRAINING COMPLETE"
echo "  Final submissions in: $BASE_DIR/outputs/submissions/"
echo "=============================================="
