#!/bin/bash
# Staged private-test inference. Waits for any train.py to finish first.
# Stage 1: all checkpoints, no TTA  (~8.5h) -> submission v1
# Stage 2: all checkpoints, 2-view TTA (~17h) -> submission v2
set -u
cd ~/freuid_challenge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vlm_distill
export HF_HUB_OFFLINE=1

PRIV_DIR=~/freuid_challenge/data/private_test/private_test
echo "[$(date)] waiting for private test data (need 134997 images)..."
while true; do
    n=$(ls "$PRIV_DIR" 2>/dev/null | wc -l)
    [ "$n" -ge 134997 ] && break
    echo "[$(date)] have $n/134997"
    sleep 300
done
echo "[$(date)] data complete - starting stage 1 (no TTA)"

python ensemble_inference.py \
    --models convnext_small eva02 --folds all \
    --test_subdir private_test/private_test --agg mean \
    > logs/private_stage1.log 2>&1
echo "[$(date)] stage 1 done - building submission v1"
python build_full_submission.py --alpha 0.5 \
    --out outputs/predictions/submission_FULL_v1_noTTA.csv \
    > logs/build_v1.log 2>&1

echo "[$(date)] starting stage 2 (2-view TTA)"
python ensemble_inference.py \
    --models convnext_small eva02 --folds all \
    --test_subdir private_test/private_test --agg mean --tta --n_tta 2 \
    > logs/private_stage2.log 2>&1
echo "[$(date)] stage 2 done - building submission v2"
python build_full_submission.py --alpha 0.5 \
    --out outputs/predictions/submission_FULL_v2_tta2.csv \
    > logs/build_v2.log 2>&1

echo "[$(date)] ALL DONE"
