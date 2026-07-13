#!/bin/bash
# =============================================================
# DEPLOY: Copy competition code to GPU server
# Run this from your LOCAL Windows machine (Git Bash / WSL)
# =============================================================

SERVER="teaching@172.18.40.113"
REMOTE_DIR="~/freuid_challenge"

echo "Deploying FREUID Challenge code to $SERVER:$REMOTE_DIR"

# Create remote directory
ssh $SERVER "mkdir -p $REMOTE_DIR/logs"

# Copy all Python files
scp -r \
    config.py \
    dataset.py \
    augmentations.py \
    forensics.py \
    models.py \
    metrics.py \
    train.py \
    inference.py \
    ensemble.py \
    eda.py \
    setup_server.sh \
    run_baseline.sh \
    run_full_ensemble.sh \
    $SERVER:$REMOTE_DIR/

echo "Done! Files deployed."
echo ""
echo "Now SSH into the server and run:"
echo "  ssh $SERVER"
echo "  cd $REMOTE_DIR"
echo "  conda activate vlm_distill"
echo "  bash setup_server.sh    # Download data"
echo "  bash run_baseline.sh    # Train first model"
