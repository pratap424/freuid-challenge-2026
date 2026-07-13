#!/bin/bash
# =============================================================
# FREUID Challenge 2026 - Server Setup Script
# Run this on dslab server (172.18.40.113) as user 'teaching'
# =============================================================

set -e

# Activate environment
source activate vlm_distill 2>/dev/null || conda activate vlm_distill

echo "=== Step 1: Check GPU ==="
nvidia-smi
echo ""

echo "=== Step 2: Check Python/PyTorch ==="
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB' if torch.cuda.is_available() else '')"
echo ""

echo "=== Step 3: Install competition dependencies ==="
pip install --upgrade pip
pip install timm albumentations opencv-python-headless scikit-learn pandas \
    scipy lightgbm matplotlib seaborn tqdm kaggle pillow --quiet

echo ""
echo "=== Step 4: Setup Kaggle API ==="
# Make sure ~/.kaggle/kaggle.json exists with your API key
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "ERROR: Kaggle API key not found!"
    echo "1. Go to https://www.kaggle.com/settings"
    echo "2. Click 'Create New Token' under API section"
    echo "3. Place kaggle.json in ~/.kaggle/"
    echo "4. Run: chmod 600 ~/.kaggle/kaggle.json"
    exit 1
fi
chmod 600 ~/.kaggle/kaggle.json

echo ""
echo "=== Step 5: Download competition data ==="
mkdir -p ~/freuid_challenge/data
cd ~/freuid_challenge/data

# Accept competition rules on Kaggle website first!
kaggle competitions download -c the-freuid-challenge-2026-ijcai-ecai

echo ""
echo "=== Step 6: Extract data ==="
# Extract whatever format it comes in
for f in *.zip; do
    [ -f "$f" ] && unzip -o "$f" -d . && echo "Extracted $f"
done

echo ""
echo "=== Step 7: Show data structure ==="
echo "--- Files ---"
ls -la
echo ""
echo "--- Directory tree (depth 2) ---"
find . -maxdepth 2 -type f | head -50
echo ""
echo "--- CSV previews ---"
for csv in *.csv; do
    [ -f "$csv" ] && echo "=== $csv ===" && head -5 "$csv" && echo "Lines: $(wc -l < $csv)" && echo ""
done

echo ""
echo "=== Step 8: Count images ==="
echo "Train images: $(find train* -name '*.jpg' -o -name '*.png' -o -name '*.jpeg' 2>/dev/null | wc -l)"
echo "Test images: $(find test* -name '*.jpg' -o -name '*.png' -o -name '*.jpeg' 2>/dev/null | wc -l)"

echo ""
echo "=== SETUP COMPLETE ==="
echo "Working directory: ~/freuid_challenge/"
echo "Data directory: ~/freuid_challenge/data/"
