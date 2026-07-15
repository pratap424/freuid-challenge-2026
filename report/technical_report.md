# Ensemble-Based Identity Document Fraud Detection for the FREUID Challenge 2026

**Team:** Yash (pratap424) · IIT Mandi
**Code:** https://github.com/pratap424/freuid-challenge-2026
**Weights:** https://www.kaggle.com/datasets/ypsrathore/freuid-2026-pratap424-weights

## 1. Approach overview

We treat fraud detection as binary image classification with an ensemble designed to
maximize recall of fraudulent documents at the strict APCER@1%BPCER operating point of
the FREUID metric.

**Models (all trained June 30 – July 2, 2026, before the code freeze):**

| Backbone | Input | Folds | Pretraining |
|---|---|---|---|
| ConvNeXt-Small (`fb_in22k_ft_in1k_384`) | 384px | 5 | ImageNet-22k→1k |
| EVA02-Large (`mim_m38m_ft_in22k_in1k`) | 448px | 1 (fold 0) | MIM M38M |

An EfficientNet-B3 (5 folds) was also trained but **excluded** after out-of-fold analysis
showed it degraded the ensemble (OOF FREUID 0.107 vs 0.00008 for ConvNeXt; public LB
confirmed: 0.288 alone).

## 2. Training

- Stratified 5-fold CV by `label × document type` (10 strata)
- Loss: 0.5·BCE + 0.5·Focal (γ=2, α=0.75), label smoothing 0.1
- AdamW, cosine schedule with 2-epoch warmup, AMP, EMA (decay 0.9998)
- Mixup (α=0.4) / CutMix (α=1.0) at p=0.5
- Heavy augmentation targeting capture-condition shift: geometric (RRC, flip,
  shift-scale-rotate, perspective), photometric (color jitter, brightness/contrast),
  degradation (Gaussian noise, JPEG compression 30–95, downscale 0.4–0.9, blur),
  occlusion (coarse dropout). A silent albumentations-v2 API incompatibility initially
  disabled 4 of these; fixing it improved public LB from 0.144 to 0.133 (single fold).

## 3. Inference and ensembling

- 8-view TTA on public test; staged TTA on private test (identity + horizontal flip)
- All model×fold×TTA predictions aggregated as: **α·mean + (1−α)·max** with α=0.5.
  The max term preserves fraud evidence found by any single ensemble member — at the
  1% BPCER operating point, a missed fraud is far costlier than a slightly elevated
  bona-fide score. Public LB validated this: α=0.5 (0.07894) > α=0.7 (0.07955) >
  plain mean (0.08022) > power-mean p=3 (0.07974) ≫ rank averaging (0.12695).
- Score calibration (isotonic/Platt) was evaluated and rejected: both FREUID components
  are ranking-based, so monotonic recalibration cannot change the score.

## 4. Results

| Submission | Public LB |
|---|---|
| EfficientNet-B3 f0 (excluded model) | 0.28818 |
| ConvNeXt-Small f0 | 0.13277 |
| ConvNeXt-Small 5-fold | 0.10950 |
| + TTA | 0.09786 |
| + EVA02-Large f0 | 0.08237 |
| + 8-view TTA everywhere, maxblend α=0.7 | 0.07955 |
| **+ maxblend α=0.5 (final public config)** | **0.07894** |
| Full submission incl. real private predictions (v1, v2) | 0.07894 public; private hidden |

Final submissions predict all 142,818 test rows with model outputs (no placeholder
values). Private-test inference used the identical frozen weights with staged TTA
(1, 2, then 8 views).

## 5. Code-freeze compliance

All weights predate the July 13 private-test release (training logs and file
timestamps available for audit). Post-freeze work consisted exclusively of:
inference on private images with unchanged weights, ensemble aggregation,
documentation, and packaging. A fold-1 EVA02 run accidentally launched on July 13
was terminated mid-run and its checkpoint deleted; it contributes to nothing.

## 6. Reproducibility

- `README.md` — full pipeline instructions
- `Dockerfile` — pinned environment
- Hardware: single NVIDIA RTX A5000 (24GB); total training ≈ 60 GPU-hours
