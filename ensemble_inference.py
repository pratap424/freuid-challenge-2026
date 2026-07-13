"""
FREUID Challenge 2026 - Advanced Ensemble Inference
Combines multiple models × multiple folds × TTA with smart aggregation.
Key insight: APCER@1%BPCER is destroyed by even a few fraud images scored
as bonafide. Power-mean ensembling preserves strong fraud signals.
"""
import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast
from tqdm import tqdm
from scipy.stats import rankdata

from config import Config, MODEL_CONFIGS
from models import build_model
from dataset import FREUIDDataset
from augmentations import get_valid_transforms, get_tta_transforms


# ==================== Aggregation Methods ====================

def arithmetic_mean(preds_list, weights=None):
    """Standard arithmetic mean."""
    if weights is None:
        return np.mean(preds_list, axis=0)
    weights = np.array(weights) / np.sum(weights)
    return np.sum([w * p for w, p in zip(weights, preds_list)], axis=0)


def rank_average(preds_list):
    """
    Rank averaging: convert to ranks, average, convert back.
    More robust than probability averaging when models have different calibration.
    """
    n = len(preds_list[0])
    rank_sum = np.zeros(n)
    for preds in preds_list:
        ranks = rankdata(preds) / n  # Normalize to [0, 1]
        rank_sum += ranks
    return rank_sum / len(preds_list)


def power_mean(preds_list, p=3):
    """
    Power mean with p>1 preserves strong fraud signals.
    Key for APCER@1%BPCER: if ANY model says fraud, keep that signal.
    
    p=1: arithmetic mean (standard)
    p=2: RMS (mild fraud preservation)
    p=3: strong fraud preservation (recommended for FREUID)
    """
    eps = 1e-8
    preds_array = np.array(preds_list)
    preds_clipped = np.clip(preds_array, eps, 1 - eps)
    
    powered = np.power(preds_clipped, p)
    mean_powered = np.mean(powered, axis=0)
    result = np.power(mean_powered, 1.0 / p)
    
    return np.clip(result, 0, 1)


def max_blend(preds_list, alpha=0.7):
    """
    Blend arithmetic mean with max.
    alpha * mean + (1-alpha) * max
    Preserves fraud signals while maintaining smooth ranking.
    """
    mean_preds = np.mean(preds_list, axis=0)
    max_preds = np.max(preds_list, axis=0)
    return alpha * mean_preds + (1 - alpha) * max_preds


# ==================== Prediction ====================

@torch.no_grad()
def predict_single(model, loader, config):
    """Predict with a single model."""
    model.eval()
    all_preds = []
    all_ids = []
    
    for batch in tqdm(loader, desc="Predicting", ncols=100, leave=False):
        images = batch['image'].cuda(non_blocking=True)
        
        with autocast('cuda', enabled=config.use_amp):
            logits = model(images)
        
        preds = torch.sigmoid(logits).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_ids.extend(batch['image_id'])
    
    return np.array(all_preds), all_ids


@torch.no_grad()
def predict_tta(model, test_df, config, n_tta=8):
    """Predict with Test-Time Augmentation."""
    tta_transforms = get_tta_transforms(config.img_size)
    n_tta = min(n_tta, len(tta_transforms))
    
    all_tta_preds = []
    
    for i in range(n_tta):
        dataset = FREUIDDataset(
            df=test_df,
            data_dir=config.data_dir,
            image_col=config.image_col,
            transform=tta_transforms[i],
            is_test=True,
        )
        loader = DataLoader(
            dataset, batch_size=config.batch_size * 2,
            shuffle=False, num_workers=config.num_workers,
            pin_memory=True, drop_last=False,
        )
        
        preds, ids = predict_single(model, loader, config)
        all_tta_preds.append(preds)
        print(f"  TTA {i+1}/{n_tta}: mean={preds.mean():.4f}, std={preds.std():.4f}")
    
    # Average TTA predictions
    avg_preds = np.mean(all_tta_preds, axis=0)
    return avg_preds, ids


def load_model(config, fold, use_ema=True):
    """Load model checkpoint."""
    config.pretrained = False  # weights come from the checkpoint; avoids HF download
    model = build_model(config)
    
    ckpt_dir = os.path.join(config.checkpoint_dir, config.experiment_name)
    ckpt_path = os.path.join(ckpt_dir, f'best_fold{fold}.pth')
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    if use_ema and ckpt.get('ema_state_dict') is not None:
        model.load_state_dict(ckpt['ema_state_dict'])
        print(f"  Loaded EMA model from fold {fold} (epoch {ckpt.get('epoch', '?')}, score={ckpt.get('best_score', 0):.4f})")
    else:
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"  Loaded model from fold {fold} (epoch {ckpt.get('epoch', '?')})")
    
    model = model.cuda()
    model.eval()
    return model


def get_test_df(config):
    """Load test dataframe, filtering to images that exist on disk."""
    test_csv_path = config.test_csv_path
    
    if os.path.exists(test_csv_path):
        full_test_df = pd.read_csv(test_csv_path)
    else:
        test_dir = config.test_images_path
        files = os.listdir(test_dir)
        ids = [os.path.splitext(f)[0] for f in sorted(files)]
        full_test_df = pd.DataFrame({config.image_col: ids})
    
    # Filter to existing images
    test_dir = config.test_images_path
    if os.path.isdir(test_dir):
        existing_files = set(os.path.splitext(f)[0] for f in os.listdir(test_dir))
        test_df = full_test_df[full_test_df[config.image_col].isin(existing_files)].reset_index(drop=True)
        n_missing = len(full_test_df) - len(test_df)
        print(f"Total submission IDs: {len(full_test_df)}")
        print(f"Images on disk: {len(test_df)} (public test)")
        print(f"Missing (private test): {n_missing}")
    else:
        test_df = full_test_df
    
    return test_df, full_test_df


# ==================== Main Ensemble ====================

def ensemble_inference(models_to_run, data_dir=None, use_tta=False, n_tta=8,
                       agg_method='power3', test_subdir=None):
    """
    Run ensemble inference across multiple models and folds.

    agg_method: 'mean', 'rank', 'power2', 'power3', 'maxblend'
    test_subdir: override test images dir (e.g. 'private_test/private_test')
    """
    all_individual_preds = []  # Every individual prediction (model × fold)
    all_model_preds = []       # Per-model averaged predictions
    raw_tag = 'priv_' if (test_subdir and 'private' in test_subdir) else ''

    # Get test data
    first_config = MODEL_CONFIGS[models_to_run[0][0]](0)
    if data_dir:
        first_config.data_dir = data_dir
    if test_subdir:
        first_config.test_images_dir = test_subdir
    test_df, full_test_df = get_test_df(first_config)
    
    ids = None
    
    for model_name, folds in models_to_run:
        print(f"\n{'='*60}")
        print(f"  Model: {model_name}")
        print(f"{'='*60}")
        
        fold_preds = []
        
        for fold in folds:
            print(f"\n--- Fold {fold} ---")
            config = MODEL_CONFIGS[model_name](fold)
            if data_dir:
                config.data_dir = data_dir
            if test_subdir:
                config.test_images_dir = test_subdir
            
            try:
                model = load_model(config, fold)
            except FileNotFoundError as e:
                print(f"  Skipping: {e}")
                continue
            
            if use_tta:
                preds, ids = predict_tta(model, test_df, config, n_tta)
            else:
                dataset = FREUIDDataset(
                    df=test_df,
                    data_dir=config.data_dir,
                    image_col=config.image_col,
                    transform=get_valid_transforms(config.img_size),
                    is_test=True,
                )
                loader = DataLoader(
                    dataset, batch_size=config.batch_size * 2,
                    shuffle=False, num_workers=config.num_workers,
                    pin_memory=True, drop_last=False,
                )
                preds, ids = predict_single(model, loader, config)
            
            fold_preds.append(preds)
            all_individual_preds.append(preds)
            print(f"  Preds: mean={preds.mean():.4f}, std={preds.std():.4f}")

            # Save raw per-fold predictions so any blend can be rebuilt offline
            raw_dir = os.path.join('outputs', 'predictions', 'raw')
            os.makedirs(raw_dir, exist_ok=True)
            tta_tag = '_tta' if use_tta else ''
            np.save(os.path.join(raw_dir, f'{raw_tag}{model_name}_f{fold}{tta_tag}.npy'), preds)
            pd.Series(ids).to_csv(os.path.join(raw_dir, f'{raw_tag}test_ids.csv'), index=False, header=['id'])
            
            del model
            torch.cuda.empty_cache()
        
        if fold_preds:
            avg_fold_preds = np.mean(fold_preds, axis=0)
            all_model_preds.append(avg_fold_preds)
            print(f"\n  {model_name} ensemble ({len(fold_preds)} folds): "
                  f"mean={avg_fold_preds.mean():.4f}, std={avg_fold_preds.std():.4f}")
    
    if not all_individual_preds:
        raise RuntimeError("No predictions generated!")
    
    # ==================== Multiple Aggregation Methods ====================
    print(f"\n{'='*60}")
    print(f"  AGGREGATION ({len(all_individual_preds)} total predictions)")
    print(f"{'='*60}")
    
    results = {}
    
    # 1. Arithmetic mean (baseline)
    mean_preds = arithmetic_mean(all_individual_preds)
    results['mean'] = mean_preds
    print(f"  Arithmetic mean: mean={mean_preds.mean():.4f}, std={mean_preds.std():.4f}")
    
    # 2. Rank average
    rank_preds = rank_average(all_individual_preds)
    results['rank'] = rank_preds
    print(f"  Rank average:    mean={rank_preds.mean():.4f}, std={rank_preds.std():.4f}")
    
    # 3. Power mean p=2
    pm2_preds = power_mean(all_individual_preds, p=2)
    results['power2'] = pm2_preds
    print(f"  Power mean p=2:  mean={pm2_preds.mean():.4f}, std={pm2_preds.std():.4f}")
    
    # 4. Power mean p=3
    pm3_preds = power_mean(all_individual_preds, p=3)
    results['power3'] = pm3_preds
    print(f"  Power mean p=3:  mean={pm3_preds.mean():.4f}, std={pm3_preds.std():.4f}")
    
    # 5. Max blend
    mb_preds = max_blend(all_individual_preds, alpha=0.7)
    results['maxblend'] = mb_preds
    print(f"  Max blend:       mean={mb_preds.mean():.4f}, std={mb_preds.std():.4f}")
    
    # Select the requested method
    final_preds = results.get(agg_method, mean_preds)
    
    print(f"\n  SELECTED: {agg_method}")
    print(f"  Final preds: mean={final_preds.mean():.4f}, std={final_preds.std():.4f}")
    
    # Build ALL submissions (one per method)
    out_dir = os.path.join('outputs', 'predictions')
    os.makedirs(out_dir, exist_ok=True)
    
    for method_name, method_preds in results.items():
        pred_df = pd.DataFrame({'id': ids, 'label': method_preds})
        sub_df = full_test_df[['id']].merge(pred_df, on='id', how='left')
        sub_df['label'] = sub_df['label'].fillna(0.5)
        
        model_str = '_'.join([m for m, _ in models_to_run])
        out_path = os.path.join(out_dir, f'sub_{model_str}_{method_name}.csv')
        sub_df.to_csv(out_path, index=False)
        print(f"  Saved {method_name} → {out_path}")
    
    # Return the selected method's submission
    pred_df = pd.DataFrame({'id': ids, 'label': final_preds})
    sub_df = full_test_df[['id']].merge(pred_df, on='id', how='left')
    sub_df['label'] = sub_df['label'].fillna(0.5)
    
    n_predicted = (sub_df['label'] != 0.5).sum()
    n_default = (sub_df['label'] == 0.5).sum()
    print(f"\nSubmission: {len(sub_df)} rows ({n_predicted} predicted, {n_default} default)")
    
    return sub_df, results


def main():
    parser = argparse.ArgumentParser(description='FREUID Advanced Ensemble Inference')
    parser.add_argument('--models', nargs='+', default=['convnext_small'],
                        help='Model names to ensemble')
    parser.add_argument('--folds', type=str, default='all',
                        help='Folds: "all" or comma-separated like "0,1,2"')
    parser.add_argument('--tta', action='store_true', help='Use TTA')
    parser.add_argument('--n_tta', type=int, default=8, help='Number of TTA views')
    parser.add_argument('--agg', type=str, default='power3',
                        choices=['mean', 'rank', 'power2', 'power3', 'maxblend'],
                        help='Aggregation method')
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--test_subdir', type=str, default=None,
                        help="Override test images dir, e.g. 'private_test/private_test'")
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()
    
    # Parse folds
    if args.folds == 'all':
        folds = list(range(5))
    else:
        folds = [int(f) for f in args.folds.split(',')]
    
    models_to_run = [(m, folds) for m in args.models]
    
    sub_df, results = ensemble_inference(
        models_to_run,
        data_dir=args.data_dir,
        use_tta=args.tta,
        n_tta=args.n_tta,
        agg_method=args.agg,
        test_subdir=args.test_subdir,
    )
    
    # Save main submission
    out_dir = os.path.join('outputs', 'predictions')
    os.makedirs(out_dir, exist_ok=True)
    
    if args.output:
        out_path = args.output
    else:
        model_str = '_'.join(args.models)
        tta_str = '_tta' if args.tta else ''
        out_path = os.path.join(out_dir, f'submission_{model_str}_{args.agg}{tta_str}.csv')
    
    sub_df.to_csv(out_path, index=False)
    print(f"\nMain submission saved to {out_path}")


if __name__ == '__main__':
    main()
