"""
FREUID Challenge 2026 - Inference Script
Generate predictions for test set with optional TTA.

Usage:
    python inference.py --model eva02 --fold 0
    python inference.py --model eva02 --fold all --tta
"""
import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast
from tqdm import tqdm

from config import Config, MODEL_CONFIGS
from dataset import FREUIDDataset
from augmentations import get_valid_transforms, get_tta_transforms
from models import build_model


@torch.no_grad()
def predict(model, loader, config):
    """Generate predictions for a dataloader."""
    model.eval()
    
    all_preds = []
    all_ids = []
    
    for batch in tqdm(loader, desc="Predicting", ncols=100):
        images = batch['image'].cuda(non_blocking=True)
        
        with autocast('cuda', enabled=config.use_amp):
            logits = model(images)
        
        preds = torch.sigmoid(logits).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_ids.extend(batch['image_id'])
    
    return np.array(all_preds), all_ids


@torch.no_grad()
def predict_tta(model, test_df, config, tta_transforms):
    """Generate predictions with Test-Time Augmentation."""
    model.eval()
    
    all_tta_preds = []
    
    for tta_idx, tta_transform in enumerate(tta_transforms):
        print(f"  TTA {tta_idx + 1}/{len(tta_transforms)}")
        
        dataset = FREUIDDataset(
            df=test_df,
            data_dir=config.data_dir,
            transform=tta_transform,
            is_test=True,
            use_ela=config.use_ela,
            image_col=config.image_col,
        )
        
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size * 2,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
        )
        
        preds, ids = predict(model, loader, config)
        all_tta_preds.append(preds)
    
    # Average TTA predictions
    avg_preds = np.mean(all_tta_preds, axis=0)
    return avg_preds, ids


def load_model(config, fold, use_ema=True):
    """Load trained model checkpoint."""
    model = build_model(config)
    
    ckpt_path = os.path.join(
        config.checkpoint_dir, config.experiment_name, f'best_fold{fold}.pth'
    )
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    if use_ema and ckpt.get('ema_state_dict') is not None:
        model.load_state_dict(ckpt['ema_state_dict'])
        print(f"Loaded EMA model from fold {fold} (epoch {ckpt['epoch']}, score={ckpt['best_score']:.4f})")
    else:
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded model from fold {fold} (epoch {ckpt['epoch']}, score={ckpt['best_score']:.4f})")
    
    model = model.cuda()
    model.eval()
    return model


def inference_single_model(args):
    """Run inference for a single model (possibly multiple folds)."""
    config_fn = MODEL_CONFIGS[args.model]
    folds = list(range(5)) if args.fold == 'all' else [int(args.fold)]
    
    # Load test data
    config = config_fn(0)
    if args.data_dir:
        config.data_dir = args.data_dir
    
    test_csv_path = config.test_csv_path
    
    if os.path.exists(test_csv_path):
        full_test_df = pd.read_csv(test_csv_path)
    else:
        test_dir = config.test_images_path
        files = os.listdir(test_dir)
        ids = [os.path.splitext(f)[0] for f in sorted(files)]
        full_test_df = pd.DataFrame({config.image_col: ids})
    
    # Filter to only IDs with existing images on disk
    test_dir = config.test_images_path
    if os.path.isdir(test_dir):
        existing_files = set(os.path.splitext(f)[0] for f in os.listdir(test_dir))
        test_df = full_test_df[full_test_df[config.image_col].isin(existing_files)].reset_index(drop=True)
        n_missing = len(full_test_df) - len(test_df)
        print(f"Total submission IDs: {len(full_test_df)}")
        print(f"Images on disk: {len(test_df)} (public test)")
        print(f"Missing (private test): {n_missing} → will fill with 0.5")
    else:
        test_df = full_test_df
        print(f"Test samples: {len(test_df)}")
    
    # Predict per fold
    fold_preds = []
    
    for fold in folds:
        print(f"\n--- Fold {fold} ---")
        config = config_fn(fold)
        if args.data_dir:
            config.data_dir = args.data_dir
        
        model = load_model(config, fold, use_ema=True)
        
        if args.tta:
            tta_tfms = get_tta_transforms(config.img_size)
            if args.tta_n:
                tta_tfms = tta_tfms[:args.tta_n]
            preds, ids = predict_tta(model, test_df, config, tta_tfms)
        else:
            dataset = FREUIDDataset(
                df=test_df,
                data_dir=config.data_dir,
                transform=get_valid_transforms(config.img_size),
                is_test=True,
                use_ela=config.use_ela,
                image_col=config.image_col,
            )
            loader = DataLoader(
                dataset,
                batch_size=config.batch_size * 2,
                shuffle=False,
                num_workers=config.num_workers,
                pin_memory=True,
            )
            preds, ids = predict(model, loader, config)
        
        fold_preds.append(preds)
        
        # Clean up
        del model
        torch.cuda.empty_cache()
    
    # Average across folds
    avg_preds = np.mean(fold_preds, axis=0)
    
    # Build predictions for images we could score
    pred_df = pd.DataFrame({
        'id': ids,
        'label': avg_preds,
    })
    
    # Merge with full submission (fill private test IDs with 0.5)
    sub_df = full_test_df[['id']].merge(pred_df, on='id', how='left')
    sub_df['label'] = sub_df['label'].fillna(0.5)
    
    print(f"\nSubmission: {len(sub_df)} rows ({(sub_df['label'] != 0.5).sum()} predicted, {(sub_df['label'] == 0.5).sum()} default)")
    
    out_dir = os.path.join(config.output_dir, 'predictions')
    os.makedirs(out_dir, exist_ok=True)
    
    suffix = '_tta' if args.tta else ''
    fold_str = 'all' if args.fold == 'all' else f'f{args.fold}'
    
    # Save submission
    sub_path = os.path.join(out_dir, f'submission_{args.model}_{fold_str}{suffix}.csv')
    sub_df.to_csv(sub_path, index=False)
    print(f"Saved submission to {sub_path}")
    
    return sub_df


def main():
    parser = argparse.ArgumentParser(description='FREUID Challenge Inference')
    parser.add_argument('--model', type=str, required=True,
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument('--fold', type=str, default='all',
                        help='Fold number or "all"')
    parser.add_argument('--tta', action='store_true', default=False,
                        help='Enable Test-Time Augmentation')
    parser.add_argument('--tta_n', type=int, default=None,
                        help='Number of TTA transforms (default: all 8)')
    parser.add_argument('--data_dir', type=str, default=None)
    args = parser.parse_args()
    
    inference_single_model(args)


if __name__ == '__main__':
    main()
