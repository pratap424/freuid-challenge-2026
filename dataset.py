"""
FREUID Challenge 2026 - Dataset
Custom PyTorch Dataset with forensic feature support.
Paths/columns matched to actual data structure from EDA.
"""
import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path

from forensics import compute_ela, compute_noise_residual


class FREUIDDataset(Dataset):
    """
    Dataset for FREUID Challenge.
    
    Data structure (from EDA):
      - Train images: data/train/train/{id}.jpeg
      - Test images:  data/public_test/public_test/{id}.jpeg
      - train_labels.csv has 'image_path' column: train/{id}.jpeg
      - IDs are 32-char hex UUIDs
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        data_dir: str,
        transform=None,
        is_test: bool = False,
        use_ela: bool = False,
        use_noise: bool = False,
        image_col: str = 'id',
        label_col: str = 'label',
        image_path_col: str = None,  # 'image_path' if available
    ):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform
        self.is_test = is_test
        self.use_ela = use_ela
        self.use_noise = use_noise
        self.image_col = image_col
        self.label_col = label_col
        self.image_path_col = image_path_col
        
        # Determine how to find images
        if self.image_path_col and self.image_path_col in df.columns:
            # CSV has a path column like "train/xxxx.jpeg"
            # The actual files are at data_dir/train/train/xxxx.jpeg 
            # (double nested), so we need to handle this
            self._use_path_col = True
        else:
            self._use_path_col = False
        
        # Verify first image exists
        self._verify_first_image()
    
    def _verify_first_image(self):
        """Check that we can find at least one image."""
        try:
            path = self._get_image_path(0)
            assert os.path.exists(path), f"First image not found: {path}"
        except Exception as e:
            print(f"WARNING: Could not verify first image: {e}")
            print(f"  data_dir={self.data_dir}")
            print(f"  use_path_col={self._use_path_col}")
            if self._use_path_col:
                print(f"  first image_path={self.df[self.image_path_col].iloc[0]}")
            else:
                print(f"  first id={self.df[self.image_col].iloc[0]}")
    
    def _get_image_path(self, idx):
        """Get full image path for a given index."""
        row = self.df.iloc[idx] if isinstance(idx, int) else idx
        
        if self._use_path_col:
            # image_path column has: "train/xxxx.jpeg"
            # Actual path is: data_dir/train/train/xxxx.jpeg (double nested)
            rel_path = row[self.image_path_col]  # e.g., "train/xxxx.jpeg"
            
            # Try direct: data_dir/train/xxxx.jpeg
            path = os.path.join(self.data_dir, rel_path)
            if os.path.exists(path):
                return path
            
            # Try double-nested: data_dir/train/train/xxxx.jpeg
            parts = rel_path.split('/')
            if len(parts) == 2:
                double_path = os.path.join(self.data_dir, parts[0], parts[0], parts[1])
                if os.path.exists(double_path):
                    return double_path
            
            # Fallback: just the filename in train/train/
            fname = os.path.basename(rel_path)
            for subdir in ['train/train', 'public_test/public_test', 'train', 'public_test']:
                fallback = os.path.join(self.data_dir, subdir, fname)
                if os.path.exists(fallback):
                    return fallback
            
            return path  # return original attempt for error message
        
        else:
            # No path column — build from ID
            image_id = str(row[self.image_col])
            
            # Try multiple patterns
            for subdir in ['private_test/private_test', 'public_test/public_test',
                           'train/train', 'public_test', 'train']:
                for ext in ['.jpeg', '.jpg', '.png']:
                    path = os.path.join(self.data_dir, subdir, image_id + ext)
                    if os.path.exists(path):
                        return path
            
            # Direct with .jpeg (most common in this dataset)
            return os.path.join(self.data_dir, 'public_test/public_test', image_id + '.jpeg')
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = str(row[self.image_col])
        
        # Load image
        image_path = self._get_image_path(idx)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Compute forensic features before augmentation (on full-res image)
        extra_channels = []
        
        if self.use_ela:
            ela = compute_ela(image, quality=90)
            extra_channels.append(ela)
        
        if self.use_noise:
            noise = compute_noise_residual(image)
            extra_channels.append(noise)
        
        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image)
            image_tensor = augmented['image']
        else:
            image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        
        # Add forensic channels (after augmentation resize)
        if extra_channels:
            h, w = image_tensor.shape[1], image_tensor.shape[2]
            forensic_tensors = []
            for ch in extra_channels:
                ch_resized = cv2.resize(ch, (w, h))
                # ELA/noise are already float32 [0,1], normalize to [-1,1]
                ch_norm = (ch_resized - 0.5) / 0.5
                forensic_tensors.append(torch.from_numpy(ch_norm).unsqueeze(0))
            image_tensor = torch.cat([image_tensor] + forensic_tensors, dim=0)
        
        # Return
        if self.is_test:
            return {
                'image': image_tensor,
                'image_id': image_id,
            }
        else:
            label = float(row[self.label_col])
            return {
                'image': image_tensor,
                'label': torch.tensor(label, dtype=torch.float32),
                'image_id': image_id,
            }


def prepare_folds(train_csv_path, n_folds=5, seed=42):
    """
    Prepare stratified k-fold splits.
    Uses compound stratification on label + type + is_digital 
    to ensure every fold has proportional representation.
    """
    from sklearn.model_selection import StratifiedKFold
    
    df = pd.read_csv(train_csv_path)
    label_col = 'label'
    
    print(f"Training data: {len(df)} samples")
    print(f"Columns: {list(df.columns)}")
    print(f"\nClass distribution:")
    print(f"  Bonafide (0): {(df[label_col]==0).sum()} ({(df[label_col]==0).mean()*100:.1f}%)")
    print(f"  Fraud    (1): {(df[label_col]==1).sum()} ({(df[label_col]==1).mean()*100:.1f}%)")
    
    # Detailed distribution if metadata columns exist
    if 'type' in df.columns:
        print(f"\nDocument type distribution:")
        for doc_type in sorted(df['type'].unique()):
            subset = df[df['type'] == doc_type]
            n_total = len(subset)
            n_fraud = (subset[label_col] == 1).sum()
            print(f"  {doc_type:20s}: {n_total:>6d} total | {n_fraud:>5d} fraud ({n_fraud/n_total*100:.1f}%)")
    
    if 'is_digital' in df.columns:
        print(f"\nCapture mode distribution:")
        for mode in [True, False]:
            subset = df[df['is_digital'] == mode]
            n_total = len(subset)
            n_fraud = (subset[label_col] == 1).sum()
            mode_str = "Digital" if mode else "Physical"
            print(f"  {mode_str:20s}: {n_total:>6d} total | {n_fraud:>5d} fraud ({n_fraud/n_total*100:.1f}%)")
    
    # Cross-tabulation: type x is_digital x label
    if 'type' in df.columns and 'is_digital' in df.columns:
        print(f"\n{'='*70}")
        print(f"  CROSS-TABULATION: type × is_digital × label")
        print(f"{'='*70}")
        for doc_type in sorted(df['type'].unique()):
            for is_dig in [True, False]:
                subset = df[(df['type'] == doc_type) & (df['is_digital'] == is_dig)]
                if len(subset) > 0:
                    n_fraud = (subset[label_col] == 1).sum()
                    n_bona = (subset[label_col] == 0).sum()
                    mode = "DIG" if is_dig else "PHY"
                    print(f"  {doc_type:20s} {mode}: {len(subset):>5d} ({n_bona:>5d} bona, {n_fraud:>5d} fraud = {n_fraud/len(subset)*100:.1f}%)")
    
    # Build stratification key
    # NOTE: is_digital is useless — 99.97% are digital (only 20 physical samples!)
    # Use label × type only → 10 strata, all with 5000+ samples
    if 'type' in df.columns:
        strat_key = df[label_col].astype(str) + '_' + df['type']
        print(f"\nUsing stratification: label × type ({strat_key.nunique()} strata)")
        print(f"  (Skipping is_digital — only 20 physical samples in entire dataset)")
    else:
        strat_key = df[label_col]
    
    # Create folds
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    df['fold'] = -1
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df, strat_key)):
        df.loc[val_idx, 'fold'] = fold_idx
    
    print(f"\nFold distribution:")
    for f in range(n_folds):
        fold_df = df[df['fold'] == f]
        n_pos = (fold_df[label_col] == 1).sum()
        n_neg = (fold_df[label_col] == 0).sum()
        print(f"  Fold {f}: {len(fold_df)} samples ({n_neg} bonafide, {n_pos} fraud)")
    
    return df, label_col
