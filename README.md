# FREUID Challenge 2026 — Team Yash Solution

Identity-document fraud detection for [The FREUID Challenge 2026 (IJCAI-ECAI)](https://www.kaggle.com/competitions/the-freuid-challenge-2026-ijcai-ecai).

## Approach

Ensemble of CNN + ViT classifiers trained with heavy capture-condition augmentation, combined with test-time augmentation (TTA) and fraud-preserving aggregation:

- **ConvNeXt-Small** (`convnext_small.fb_in22k_ft_in1k_384`), 384px, 5-fold CV
- **EVA02-Large** (`eva02_large_patch14_448.mim_m38m_ft_in22k_in1k`), 448px
- 8-view TTA (flips / rotations / scale / brightness)
- Aggregation: blend of arithmetic mean and element-wise max over all model×fold×TTA predictions (`0.7·mean + 0.3·max`) — the max term preserves fraud signals found by any single model, which matters for APCER@1%BPCER
- Loss: BCE + Focal composite, label smoothing 0.1, EMA weights (decay 0.9998), mixup/cutmix
- Stratified 5-fold CV by `label × document type`

## Repository layout

| File | Purpose |
|---|---|
| `config.py` | All hyperparameters + per-model presets |
| `dataset.py` | PyTorch dataset (handles competition CSV/dir layout) |
| `augmentations.py` | Train/val/TTA transforms (albumentations 2.x API) |
| `models.py` | timm backbone + GeM pooling + EMA + composite loss |
| `metrics.py` | Exact FREUID score (AuDET + APCER@1%BPCER harmonic mean) |
| `train.py` | Training loop (AMP, EMA, cosine schedule, early stopping) |
| `inference.py` | Single-model inference |
| `ensemble_inference.py` | Multi-model × multi-fold × TTA ensemble + aggregation |
| `forensics.py` | ELA / DCT / SRM forensic feature extraction (optional channel) |
| `eda.py` | Dataset exploration |

## Reproduce

### 1. Environment

```bash
docker build -t freuid .
# or: pip install torch timm albumentations opencv-python-headless scikit-learn pandas scipy tqdm
```

### 2. Data

Download competition data and extract into `data/` (expects `data/train/train/*.jpeg`,
`data/public_test/public_test/*.jpeg`, `data/train_labels.csv`, `data/sample_submission.csv`).

### 3. Train

```bash
# ConvNeXt-Small, all 5 folds (~4h/fold on RTX A5000 24GB)
python train.py --model convnext_small --fold all --epochs 15

# EVA02-Large folds (~2h/epoch)
python train.py --model eva02 --fold 0 --epochs 10
python train.py --model eva02 --fold 1 --epochs 8
```

### 4. Predict + ensemble

```bash
python ensemble_inference.py --models convnext_small eva02 --folds all --tta --agg maxblend
```

Submissions for every aggregation method are written to `outputs/predictions/`.

## Hardware

Single NVIDIA RTX A5000 (24GB), CUDA 12.8, PyTorch 2.10, timm 1.0.27.

## Code-freeze compliance (July 13, 2026)

All model weights used for private-test predictions were trained **June 30 – July 2, 2026**,
before the private test release (July 13, 07:02 UTC). Evidence: training logs
(`logs/*.log`) and checkpoint file timestamps on the training server.

- Model weights (exact training-time checkpoints, optimizer state stripped for size):
  published as a public Kaggle dataset — see `WEIGHTS.md`.
- No training, fine-tuning, weight/architecture/hyperparameter changes after July 13.
- Post-freeze commits touch only: inference orchestration (ensemble aggregation of
  unchanged weights), private-test prediction updates, documentation, and packaging —
  as permitted by the competition rules.
- An EVA02 fold-1 training run accidentally started on July 13 was killed mid-run and
  its checkpoint deleted; it is not part of any submission.

## License

MIT — see [LICENSE](LICENSE).
