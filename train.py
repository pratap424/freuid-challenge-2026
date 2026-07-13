"""
FREUID Challenge 2026 - Training Script
Full training pipeline with mixed precision, EMA, and proper CV.

Usage:
    python train.py --model eva02 --fold 0
    python train.py --model convnextv2 --fold 0 --epochs 30
    python train.py --model eva02 --fold all  # train all folds
"""
import os
import sys
import time
import argparse
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from pathlib import Path
from tqdm import tqdm

from config import Config, MODEL_CONFIGS
from dataset import FREUIDDataset, prepare_folds
from augmentations import get_train_transforms, get_valid_transforms, mixup, cutmix, mix_criterion
from models import build_model, build_loss, ModelEMA, count_parameters
from metrics import freuid_score, detailed_metrics


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # True slows training
    torch.backends.cudnn.benchmark = True


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, config, epoch, ema=None):
    """Train for one epoch. EMA is updated per optimizer step."""
    model.train()
    
    total_loss = 0
    num_batches = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(loader, desc=f"Train Epoch {epoch}", ncols=120)
    
    optimizer.zero_grad()
    
    for step, batch in enumerate(pbar):
        images = batch['image'].cuda(non_blocking=True)
        labels = batch['label'].cuda(non_blocking=True)
        
        # Mixup / CutMix
        do_mix = np.random.rand() < config.mix_prob and epoch >= config.warmup_epochs
        if do_mix:
            if np.random.rand() < 0.5:
                images, labels_a, labels_b, lam = mixup(images, labels, config.mixup_alpha)
            else:
                images, labels_a, labels_b, lam = cutmix(images, labels, config.cutmix_alpha)
        
        # Forward pass
        with autocast('cuda', enabled=config.use_amp):
            logits = model(images)
            
            if do_mix:
                loss = mix_criterion(criterion, logits, labels_a, labels_b, lam)
            else:
                loss = criterion(logits, labels)
            
            loss = loss / config.gradient_accumulation
        
        # Backward pass
        scaler.scale(loss).backward()
        
        if (step + 1) % config.gradient_accumulation == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # Update EMA after each optimizer step (CRITICAL: not per-epoch!)
            if ema is not None:
                ema.update(model)
        
        # Track metrics
        total_loss += loss.item() * config.gradient_accumulation
        num_batches += 1
        
        with torch.no_grad():
            preds = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(preds.tolist())
            if not do_mix:
                all_labels.extend(labels.cpu().numpy().tolist())
        
        pbar.set_postfix({
            'loss': f'{total_loss/num_batches:.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}',
        })
    
    if scheduler is not None:
        scheduler.step()
    
    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


@torch.no_grad()
def validate(model, loader, criterion, config):
    """Validate the model."""
    model.eval()
    
    total_loss = 0
    num_batches = 0
    all_preds = []
    all_labels = []
    all_ids = []
    
    pbar = tqdm(loader, desc="Validating", ncols=120)
    
    for batch in pbar:
        images = batch['image'].cuda(non_blocking=True)
        labels = batch['label'].cuda(non_blocking=True)
        
        with autocast('cuda', enabled=config.use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        
        total_loss += loss.item()
        num_batches += 1
        
        preds = torch.sigmoid(logits).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        all_ids.extend(batch['image_id'])
        
        pbar.set_postfix({'loss': f'{total_loss/num_batches:.4f}'})
    
    avg_loss = total_loss / max(num_batches, 1)
    
    # Compute metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Diagnostic: check if predictions are constant
    print(f"  Val preds: mean={all_preds.mean():.4f}, std={all_preds.std():.4f}, "
          f"min={all_preds.min():.4f}, max={all_preds.max():.4f}")
    
    metrics = detailed_metrics(all_labels, all_preds)
    metrics['val_loss'] = avg_loss
    
    return metrics, all_preds, all_labels, all_ids


def get_optimizer(model, config):
    """Create optimizer with differential learning rates."""
    backbone_params = list(model.get_backbone_params())
    head_params = list(model.get_head_params())
    
    param_groups = [
        {'params': backbone_params, 'lr': config.lr * config.backbone_lr_mult, 'name': 'backbone'},
        {'params': head_params, 'lr': config.lr, 'name': 'head'},
    ]
    
    if config.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(param_groups, weight_decay=config.weight_decay)
    elif config.optimizer == 'adam':
        optimizer = torch.optim.Adam(param_groups, weight_decay=config.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")
    
    return optimizer


def get_scheduler(optimizer, config, num_training_steps):
    """Create learning rate scheduler."""
    if config.scheduler == 'cosine_warmup':
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        
        warmup_scheduler = LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0,
            total_iters=config.warmup_epochs
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=config.epochs - config.warmup_epochs,
            eta_min=config.min_lr
        )
        scheduler = SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[config.warmup_epochs]
        )
    elif config.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs, eta_min=config.min_lr
        )
    else:
        scheduler = None
    
    return scheduler


def train_fold(config, fold, train_df, label_col):
    """Train a single fold."""
    print(f"\n{'='*60}")
    print(f"  TRAINING FOLD {fold} | Model: {config.model_name}")
    print(f"{'='*60}")
    
    config.fold = fold
    seed_everything(config.seed + fold)
    
    # Split data
    train_data = train_df[train_df['fold'] != fold].reset_index(drop=True)
    val_data = train_df[train_df['fold'] == fold].reset_index(drop=True)
    
    print(f"Train: {len(train_data)} samples | Val: {len(val_data)} samples")
    
    # Compute class weight
    n_pos = (train_data[label_col] == 1).sum()
    n_neg = (train_data[label_col] == 0).sum()
    config.pos_weight = n_neg / max(n_pos, 1)
    print(f"Class balance: {n_neg} neg / {n_pos} pos (pos_weight={config.pos_weight:.2f})")
    
    # Datasets — use data_dir + image_path column for correct path resolution
    train_dataset = FREUIDDataset(
        df=train_data,
        data_dir=config.data_dir,
        transform=get_train_transforms(config.img_size, config.aug_level),
        use_ela=config.use_ela,
        image_col=config.image_col,
        label_col=label_col,
        image_path_col='image_path' if 'image_path' in train_data.columns else None,
    )
    
    val_dataset = FREUIDDataset(
        df=val_data,
        data_dir=config.data_dir,
        transform=get_valid_transforms(config.img_size),
        use_ela=config.use_ela,
        image_col=config.image_col,
        label_col=label_col,
        image_path_col='image_path' if 'image_path' in val_data.columns else None,
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    
    # Model
    model = build_model(config)
    total_params, trainable_params = count_parameters(model)
    print(f"Model params: {total_params/1e6:.1f}M total, {trainable_params/1e6:.1f}M trainable")
    model = model.cuda()
    
    # Loss, optimizer, scheduler
    criterion = build_loss(config)
    optimizer = get_optimizer(model, config)
    
    num_training_steps = len(train_loader) * config.epochs
    scheduler = get_scheduler(optimizer, config, num_training_steps)
    
    # AMP scaler
    scaler = GradScaler('cuda', enabled=config.use_amp)
    
    # EMA
    ema = ModelEMA(model, decay=config.ema_decay) if config.ema_decay > 0 else None
    
    # Training loop
    best_score = float('inf')  # Lower is better for FREUID
    best_epoch = 0
    best_oof_preds = None
    best_oof_ids = None
    patience = 10
    no_improve = 0
    
    history = []
    
    for epoch in range(1, config.epochs + 1):
        t0 = time.time()
        
        # Train (pass ema for per-step updates)
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, config, epoch, ema=ema
        )
        
        # EMA is now updated inside train_one_epoch per optimizer step
        
        # Validate
        if epoch % config.val_interval == 0 or epoch == config.epochs:
            # Validate with EMA model if available
            eval_model = ema.module if ema is not None else model
            val_metrics, val_preds, val_labels, val_ids = validate(
                eval_model, val_loader, criterion, config
            )
            
            elapsed = time.time() - t0
            
            # Log
            print(f"\nEpoch {epoch}/{config.epochs} ({elapsed:.0f}s)")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_metrics['val_loss']:.4f}")
            print(f"  FREUID:     {val_metrics['freuid_score']:.4f}")
            print(f"  AuDET:      {val_metrics['audet']:.4f}")
            print(f"  APCER@1%:   {val_metrics['apcer_at_1pct_bpcer']:.4f}")
            print(f"  ROC-AUC:    {val_metrics['roc_auc']:.4f}")
            
            # Save history
            epoch_info = {
                'epoch': epoch,
                'train_loss': train_loss,
                **val_metrics,
                'time': elapsed,
            }
            history.append(epoch_info)
            
            # Check improvement (lower FREUID = better)
            if val_metrics['freuid_score'] < best_score:
                best_score = val_metrics['freuid_score']
                best_epoch = epoch
                best_oof_preds = val_preds
                best_oof_ids = val_ids
                no_improve = 0
                
                # Save checkpoint
                save_dir = os.path.join(config.checkpoint_dir, config.experiment_name)
                os.makedirs(save_dir, exist_ok=True)
                
                ckpt = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'ema_state_dict': ema.state_dict() if ema else None,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_score': best_score,
                    'config': vars(config),
                }
                torch.save(ckpt, os.path.join(save_dir, f'best_fold{fold}.pth'))
                print(f"  *** New best! FREUID={best_score:.4f} (saved)")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  Early stopping after {patience} epochs without improvement")
                    break
    
    print(f"\nFold {fold} Complete: Best FREUID={best_score:.4f} at epoch {best_epoch}")
    
    # Save OOF predictions
    oof_df = pd.DataFrame({
        'id': best_oof_ids,
        'pred': best_oof_preds,
    })
    save_dir = os.path.join(config.checkpoint_dir, config.experiment_name)
    oof_df.to_csv(os.path.join(save_dir, f'oof_fold{fold}.csv'), index=False)
    
    # Save history
    with open(os.path.join(save_dir, f'history_fold{fold}.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    return best_score, best_oof_preds, best_oof_ids


def main():
    parser = argparse.ArgumentParser(description='FREUID Challenge Training')
    parser.add_argument('--model', type=str, default='eva02',
                        choices=list(MODEL_CONFIGS.keys()),
                        help='Model architecture preset')
    parser.add_argument('--fold', type=str, default='0',
                        help='Fold number (0-4) or "all" for all folds')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--img_size', type=int, default=None)
    parser.add_argument('--aug_level', type=str, default=None,
                        choices=['light', 'medium', 'heavy'])
    parser.add_argument('--use_ela', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--data_dir', type=str, default=None)
    args = parser.parse_args()
    
    # Get config
    config_fn = MODEL_CONFIGS[args.model]
    
    folds = list(range(5)) if args.fold == 'all' else [int(args.fold)]
    
    # Override config values
    overrides = {}
    if args.epochs: overrides['epochs'] = args.epochs
    if args.batch_size: overrides['batch_size'] = args.batch_size
    if args.lr: overrides['lr'] = args.lr
    if args.img_size: overrides['img_size'] = args.img_size
    if args.aug_level: overrides['aug_level'] = args.aug_level
    if args.use_ela: overrides['use_ela'] = True
    if args.seed: overrides['seed'] = args.seed
    if args.debug: overrides['debug'] = True
    if args.data_dir: overrides['data_dir'] = args.data_dir
    
    # Prepare folds
    sample_config = config_fn(0)
    if args.data_dir:
        sample_config.data_dir = args.data_dir
    
    train_df, label_col = prepare_folds(sample_config.train_csv_path, n_folds=5, seed=args.seed)
    
    # Train each fold
    all_scores = []
    for fold in folds:
        config = config_fn(fold)
        for k, v in overrides.items():
            setattr(config, k, v)
        
        if args.debug:
            config.epochs = 2
            config.batch_size = 4
        
        score, _, _ = train_fold(config, fold, train_df, label_col)
        all_scores.append(score)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*60}")
    for fold, score in zip(folds, all_scores):
        print(f"  Fold {fold}: FREUID = {score:.4f}")
    if len(all_scores) > 1:
        print(f"  Mean:   FREUID = {np.mean(all_scores):.4f}")
        print(f"  Std:    FREUID = {np.std(all_scores):.4f}")


if __name__ == '__main__':
    main()
