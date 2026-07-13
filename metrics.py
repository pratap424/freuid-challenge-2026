"""
FREUID Score Metric Implementation
Matches the exact Kaggle competition metric.
"""
import numpy as np
from sklearn.metrics import roc_curve


def compute_det_curve(y_true, y_score):
    """
    Compute Detection Error Tradeoff (DET) curve.
    Returns (fpr, fnr) where:
      fpr = BPCER (bona-fide incorrectly rejected as fraud)
      fnr = APCER (attacks incorrectly accepted as bona-fide)
    
    Convention: label 0 = bona-fide, label 1 = attack/fraud
    Higher scores = more likely fraud.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1 - tpr
    return fpr, fnr, thresholds


def compute_audet(y_true, y_score):
    """
    Area under the DET curve.
    Lower AuDET = better model.
    """
    fpr, fnr, _ = compute_det_curve(y_true, y_score)
    # DET curve plots FNR (APCER) vs FPR (BPCER)
    # Area under this curve
    audet = np.trapz(fnr, fpr)
    return audet


def compute_apcer_at_bpcer(y_true, y_score, target_bpcer=0.01):
    """
    Compute APCER (Attack Presentation Classification Error Rate)
    at a fixed BPCER (Bona-fide Presentation Classification Error Rate).
    
    BPCER = FPR (false positive rate on bona-fide samples)
    APCER = FNR (false negative rate on attack samples)
    
    We want APCER at BPCER = 1%.
    """
    fpr, fnr, thresholds = compute_det_curve(y_true, y_score)
    
    # Find the threshold where FPR (BPCER) <= target_bpcer
    # fpr is sorted ascending, so we find where fpr <= target_bpcer
    valid_indices = np.where(fpr <= target_bpcer)[0]
    
    if len(valid_indices) == 0:
        # If no threshold achieves target BPCER, return worst case
        return 1.0
    
    # Take the last valid index (highest threshold that achieves target BPCER)
    idx = valid_indices[-1]
    
    # Interpolate if possible for more accurate estimate
    if idx < len(fpr) - 1:
        # Linear interpolation
        fpr_low, fpr_high = fpr[idx], fpr[idx + 1]
        fnr_low, fnr_high = fnr[idx], fnr[idx + 1]
        
        if fpr_high - fpr_low > 0:
            alpha = (target_bpcer - fpr_low) / (fpr_high - fpr_low)
            apcer = fnr_low + alpha * (fnr_high - fnr_low)
        else:
            apcer = fnr[idx]
    else:
        apcer = fnr[idx]
    
    return apcer


def freuid_score(y_true, y_score):
    """
    Compute the FREUID Score.
    
    FREUID = 1 - HarmonicMean(1-AuDET, 1-APCER@1%BPCER)
    
    Lower is better. Perfect score = 0.0, worst = 1.0
    
    Args:
        y_true: array of true labels (0=bona-fide, 1=fraud)
        y_score: array of fraud probability scores (higher = more likely fraud)
    
    Returns:
        float: FREUID score (lower is better)
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    
    audet = compute_audet(y_true, y_score)
    apcer = compute_apcer_at_bpcer(y_true, y_score, target_bpcer=0.01)
    
    g_audet = 1.0 - audet
    g_apcer = 1.0 - apcer
    
    if g_audet + g_apcer == 0:
        return 1.0
    
    score = 1.0 - 2.0 * g_audet * g_apcer / (g_audet + g_apcer)
    return score


def detailed_metrics(y_true, y_score):
    """
    Compute all component metrics for debugging.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    
    audet = compute_audet(y_true, y_score)
    apcer = compute_apcer_at_bpcer(y_true, y_score, target_bpcer=0.01)
    score = freuid_score(y_true, y_score)
    
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_true, y_score)
    
    return {
        'freuid_score': score,
        'audet': audet,
        'apcer_at_1pct_bpcer': apcer,
        'g_audet': 1.0 - audet,
        'g_apcer': 1.0 - apcer,
        'roc_auc': auc,
        'n_bonafide': int(np.sum(y_true == 0)),
        'n_fraud': int(np.sum(y_true == 1)),
    }


if __name__ == '__main__':
    # Quick sanity check
    np.random.seed(42)
    n = 1000
    y = np.random.binomial(1, 0.3, n)
    # Good model
    scores_good = y * 0.8 + np.random.normal(0, 0.15, n)
    scores_good = np.clip(scores_good, 0, 1)
    # Bad model 
    scores_bad = np.random.rand(n)
    
    print("=== Good Model ===")
    for k, v in detailed_metrics(y, scores_good).items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    print("\n=== Random Model ===")
    for k, v in detailed_metrics(y, scores_bad).items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
