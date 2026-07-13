#!/bin/bash
# =============================================================
# FREUID Challenge 2026 - Master Training Pipeline
# Run this on GPU server after setup_server.sh
# =============================================================

set -e

# Configuration
CONDA_ENV="vlm_distill"
BASE_DIR="$HOME/freuid_challenge"
DATA_DIR="$BASE_DIR/data"
CODE_DIR="$BASE_DIR"  # or wherever you copy the code

# Activate environment
source activate $CONDA_ENV 2>/dev/null || conda activate $CONDA_ENV

cd $CODE_DIR

echo "=============================================="
echo "  FREUID Challenge 2026 - Training Pipeline"
echo "=============================================="
echo "  Time: $(date)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "  Data: $DATA_DIR"
echo ""

# ===========================
# PHASE 1: Data Analysis
# ===========================
echo ">>> PHASE 1: Exploratory Data Analysis"
python eda.py --data_dir $DATA_DIR
echo ""

# ===========================
# PHASE 2: Quick Baseline (1 fold, 1 model)
# ===========================
echo ">>> PHASE 2: Quick Baseline - EVA02 Fold 0"
python train.py \
    --model eva02 \
    --fold 0 \
    --epochs 15 \
    --aug_level medium \
    --data_dir $DATA_DIR

echo ""
echo ">>> Running inference for baseline..."
python inference.py \
    --model eva02 \
    --fold 0 \
    --data_dir $DATA_DIR

echo ""
echo "=============================================="
echo "  BASELINE COMPLETE"
echo "  Submit outputs/predictions/submission_eva02_f0.csv"
echo "  Then proceed to Phase 3 (full training)"
echo "=============================================="
