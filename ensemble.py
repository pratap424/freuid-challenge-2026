"""
FREUID Challenge 2026 - Ensemble & Score Calibration
Combines multiple model predictions and calibrates scores for optimal FREUID metric.

Usage:
    python ensemble.py --pred_dir outputs/predictions/ --oof_dir checkpoints/
"""
import os
import argparse
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from itertools import combinations

from metrics import freuid_score, detailed_metrics


def load_oof_predictions(oof_dir, model_names, n_folds=5):
    """
    Load out-of-fold predictions from all models.
    Returns aligned predictions and labels.
    """
    all_oof = {}
    
    for model_name in model_names:
        model_dir = os.path.join(oof_dir, model_name)
        dfs = []
        for fold in range(n_folds):
            oof_path = os.path.join(model_dir, f'oof_fold{fold}.csv')
            if os.path.exists(oof_path):
                df = pd.read_csv(oof_path)
                dfs.append(df)
            else:
                print(f"WARNING: Missing OOF for {model_name} fold {fold}")
        
        if dfs:
            oof_df = pd.concat(dfs).reset_index(drop=True)
            all_oof[model_name] = oof_df
    
    return all_oof


def load_test_predictions(pred_dir, model_names):
    """Load test predictions from all models."""
    test_preds = {}
    
    for model_name in model_names:
        # Try multiple naming patterns
        for pattern in [
            f'{model_name}_all_tta.csv',
            f'{model_name}_all.csv',
            f'submission_{model_name}_all_tta.csv',
            f'submission_{model_name}_all.csv',
        ]:
            path = os.path.join(pred_dir, pattern)
            if os.path.exists(path):
                test_preds[model_name] = pd.read_csv(path)
                print(f"Loaded test preds for {model_name}: {path}")
                break
        else:
            print(f"WARNING: No test predictions found for {model_name}")
    
    return test_preds


def optimize_weights(oof_preds_dict, labels, method='nelder-mead'):
    """
    Find optimal ensemble weights that minimize FREUID score.
    """
    model_names = list(oof_preds_dict.keys())
    preds_list = [oof_preds_dict[name] for name in model_names]
    n_models = len(preds_list)
    
    def objective(weights):
        weights = np.abs(weights) / np.sum(np.abs(weights))
        blended = sum(w * p for w, p in zip(weights, preds_list))
        return freuid_score(labels, blended)
    
    # Try multiple starting points
    best_result = None
    best_score = float('inf')
    
    for trial in range(20):
        if trial == 0:
            x0 = np.ones(n_models) / n_models
        else:
            x0 = np.random.dirichlet(np.ones(n_models))
        
        result = minimize(objective, x0, method=method,
                         options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
        
        if result.fun < best_score:
            best_score = result.fun
            best_result = result
    
    optimal_weights = np.abs(best_result.x) / np.sum(np.abs(best_result.x))
    
    print(f"\n=== Optimal Ensemble Weights (FREUID={best_score:.6f}) ===")
    for name, w in zip(model_names, optimal_weights):
        print(f"  {name}: {w:.4f}")
    
    return dict(zip(model_names, optimal_weights)), best_score


def calibrate_scores(oof_scores, oof_labels, test_scores):
    """
    Calibrate scores using Isotonic Regression for optimal FREUID metric.
    """
    # Isotonic Regression (monotonic, rank-preserving)
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
    iso.fit(oof_scores, oof_labels)
    
    calibrated_oof = iso.predict(oof_scores)
    calibrated_test = iso.predict(test_scores)
    
    # Check if calibration helped
    score_before = freuid_score(oof_labels, oof_scores)
    score_after = freuid_score(oof_labels, calibrated_oof)
    
    print(f"\n=== Score Calibration ===")
    print(f"  Before: FREUID = {score_before:.6f}")
    print(f"  After:  FREUID = {score_after:.6f}")
    print(f"  {'IMPROVED' if score_after < score_before else 'NO IMPROVEMENT'}")
    
    if score_after < score_before:
        return calibrated_test, calibrated_oof, score_after
    else:
        return test_scores, oof_scores, score_before


def power_transform_search(oof_scores, labels, test_scores):
    """
    Search for optimal power transform: score^p
    Can improve separation at critical threshold.
    """
    best_power = 1.0
    best_score = freuid_score(labels, oof_scores)
    
    for p in np.arange(0.1, 5.0, 0.02):
        transformed = oof_scores ** p
        score = freuid_score(labels, transformed)
        if score < best_score:
            best_score = score
            best_power = p
    
    print(f"\n=== Power Transform Search ===")
    print(f"  Optimal power: {best_power:.2f}")
    print(f"  FREUID score:  {best_score:.6f}")
    
    if best_power != 1.0:
        return test_scores ** best_power, best_power, best_score
    return test_scores, 1.0, best_score


def rank_average(pred_dfs, weights=None):
    """
    Rank-based averaging: convert predictions to ranks, then average.
    More robust than raw score averaging.
    """
    from scipy.stats import rankdata
    
    n_models = len(pred_dfs)
    if weights is None:
        weights = np.ones(n_models) / n_models
    
    # Get predictions
    preds = []
    for df in pred_dfs:
        p = df['label'].values
        # Convert to ranks, then normalize to [0, 1]
        ranked = rankdata(p) / len(p)
        preds.append(ranked)
    
    # Weighted average of ranks
    blended = sum(w * p for w, p in zip(weights, preds))
    
    return blended


def greedy_ensemble_selection(oof_preds_dict, labels, max_models=None):
    """
    Greedy forward selection of models for ensemble.
    Adds models one at a time, keeping only those that improve the score.
    """
    model_names = list(oof_preds_dict.keys())
    if max_models is None:
        max_models = len(model_names)
    
    selected = []
    remaining = model_names.copy()
    best_score = float('inf')
    
    print(f"\n=== Greedy Ensemble Selection ===")
    
    for round_num in range(min(max_models, len(model_names))):
        best_candidate = None
        best_candidate_score = float('inf')
        
        for candidate in remaining:
            trial = selected + [candidate]
            trial_preds = [oof_preds_dict[name] for name in trial]
            blended = np.mean(trial_preds, axis=0)
            score = freuid_score(labels, blended)
            
            if score < best_candidate_score:
                best_candidate_score = score
                best_candidate = candidate
        
        if best_candidate_score < best_score:
            best_score = best_candidate_score
            selected.append(best_candidate)
            remaining.remove(best_candidate)
            print(f"  Round {round_num+1}: Added {best_candidate} → FREUID={best_score:.6f}")
        else:
            print(f"  Round {round_num+1}: No improvement, stopping")
            break
    
    print(f"\n  Selected models: {selected}")
    print(f"  Final FREUID: {best_score:.6f}")
    
    return selected, best_score


def create_submission(test_df, output_path):
    """Create properly formatted submission file."""
    sub = test_df[['id', 'label']].copy()
    sub.to_csv(output_path, index=False)
    print(f"\nSubmission saved to: {output_path}")
    print(f"  Shape: {sub.shape}")
    print(f"  Score range: [{sub['label'].min():.4f}, {sub['label'].max():.4f}]")
    print(f"  Score mean:  {sub['label'].mean():.4f}")
    return sub


def full_ensemble_pipeline(oof_dir, pred_dir, train_csv, output_dir):
    """
    Full ensemble pipeline:
    1. Load OOF + test predictions from all models
    2. Greedy model selection
    3. Weight optimization
    4. Score calibration
    5. Power transform
    6. Generate final submission
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all model directories
    model_names = [d for d in os.listdir(oof_dir)
                   if os.path.isdir(os.path.join(oof_dir, d))]
    print(f"Found models: {model_names}")
    
    # Load train labels
    train_df = pd.read_csv(train_csv)
    label_col = [c for c in train_df.columns if c in ['label', 'target', 'class']][0]
    
    # Load OOF predictions
    all_oof = load_oof_predictions(oof_dir, model_names)
    
    # Align OOF predictions with labels
    oof_preds = {}
    oof_labels = None
    for name, oof_df in all_oof.items():
        merged = oof_df.merge(train_df[['id', label_col]], on='id', how='inner')
        oof_preds[name] = merged['pred'].values
        if oof_labels is None:
            oof_labels = merged[label_col].values
    
    # Print individual model scores
    print("\n=== Individual Model Scores ===")
    for name, preds in oof_preds.items():
        m = detailed_metrics(oof_labels, preds)
        print(f"  {name}: FREUID={m['freuid_score']:.4f} | AuDET={m['audet']:.4f} | APCER@1%={m['apcer_at_1pct_bpcer']:.4f} | AUC={m['roc_auc']:.4f}")
    
    # Step 1: Greedy selection
    selected, greedy_score = greedy_ensemble_selection(oof_preds, oof_labels)
    
    # Step 2: Weight optimization on selected models
    selected_preds = {name: oof_preds[name] for name in selected}
    weights, weighted_score = optimize_weights(selected_preds, oof_labels)
    
    # Step 3: Create blended OOF predictions
    blended_oof = sum(weights[name] * oof_preds[name] for name in selected)
    
    # Step 4: Calibration
    # Load test predictions
    test_preds_dict = load_test_predictions(pred_dir, selected)
    test_ids = list(test_preds_dict.values())[0]['id'].values
    
    blended_test = sum(
        weights[name] * test_preds_dict[name]['label'].values 
        for name in selected
    )
    
    calibrated_test, calibrated_oof, cal_score = calibrate_scores(
        blended_oof, oof_labels, blended_test
    )
    
    # Step 5: Power transform
    final_test, best_power, final_oof_score = power_transform_search(
        calibrated_oof if cal_score < weighted_score else blended_oof,
        oof_labels,
        calibrated_test if cal_score < weighted_score else blended_test,
    )
    
    # Step 6: Create submission
    sub_df = pd.DataFrame({
        'id': test_ids,
        'label': final_test,
    })
    
    submission = create_submission(sub_df, os.path.join(output_dir, 'submission_final.csv'))
    
    # Also save without calibration for comparison
    sub_raw = pd.DataFrame({
        'id': test_ids,
        'label': blended_test,
    })
    create_submission(sub_raw, os.path.join(output_dir, 'submission_raw_blend.csv'))
    
    # Summary
    print(f"\n{'='*60}")
    print(f"  ENSEMBLE SUMMARY")
    print(f"{'='*60}")
    print(f"  Models used: {selected}")
    print(f"  Weights: {weights}")
    print(f"  Power transform: {best_power:.2f}")
    print(f"  Best OOF FREUID: {min(weighted_score, cal_score, final_oof_score):.6f}")
    print(f"  Submissions saved to: {output_dir}")
    
    return submission


def main():
    parser = argparse.ArgumentParser(description='FREUID Ensemble')
    parser.add_argument('--oof_dir', type=str, 
                        default=os.path.expanduser('~/freuid_challenge/checkpoints'))
    parser.add_argument('--pred_dir', type=str,
                        default=os.path.expanduser('~/freuid_challenge/outputs/predictions'))
    parser.add_argument('--train_csv', type=str,
                        default=os.path.expanduser('~/freuid_challenge/data/train.csv'))
    parser.add_argument('--output_dir', type=str,
                        default=os.path.expanduser('~/freuid_challenge/outputs/submissions'))
    args = parser.parse_args()
    
    full_ensemble_pipeline(args.oof_dir, args.pred_dir, args.train_csv, args.output_dir)


if __name__ == '__main__':
    main()
