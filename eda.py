"""
FREUID Challenge 2026 - Exploratory Data Analysis
Run this FIRST after downloading data to understand the dataset.

Usage:
    python eda.py --data_dir ~/freuid_challenge/data
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from PIL import Image
import json


def analyze_data(data_dir):
    """Comprehensive EDA of the competition data."""
    print(f"\n{'='*70}")
    print(f"  FREUID Challenge 2026 - Exploratory Data Analysis")
    print(f"  Data directory: {data_dir}")
    print(f"{'='*70}\n")
    
    # ========== 1. List all files ==========
    print("1. FILE INVENTORY")
    print("-" * 50)
    all_files = []
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            fpath = os.path.join(root, f)
            size = os.path.getsize(fpath)
            rel_path = os.path.relpath(fpath, data_dir)
            all_files.append((rel_path, size))
    
    for fpath, size in sorted(all_files)[:50]:
        size_str = f"{size/1e6:.1f}MB" if size > 1e6 else f"{size/1e3:.1f}KB" if size > 1e3 else f"{size}B"
        print(f"  {fpath:60s} {size_str}")
    
    if len(all_files) > 50:
        print(f"  ... and {len(all_files) - 50} more files")
    
    # ========== 2. Analyze CSVs ==========
    print(f"\n2. CSV FILES")
    print("-" * 50)
    csv_files = [f for f, _ in all_files if f.endswith('.csv')]
    
    csv_data = {}
    for csv_file in csv_files:
        csv_path = os.path.join(data_dir, csv_file)
        df = pd.read_csv(csv_path)
        csv_data[csv_file] = df
        
        print(f"\n  === {csv_file} ===")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Dtypes:")
        for col in df.columns:
            print(f"    {col}: {df[col].dtype} | nunique={df[col].nunique()} | nulls={df[col].isnull().sum()}")
        print(f"  First 5 rows:")
        print(df.head().to_string(index=False))
        
        # If label column exists, show distribution
        for label_col in ['label', 'target', 'class', 'is_fraud']:
            if label_col in df.columns:
                print(f"\n  Label distribution ({label_col}):")
                vc = df[label_col].value_counts()
                for val, cnt in vc.items():
                    pct = cnt / len(df) * 100
                    print(f"    {val}: {cnt:>6d} ({pct:.1f}%)")
                break
    
    # ========== 3. Analyze Images ==========
    print(f"\n3. IMAGE ANALYSIS")
    print("-" * 50)
    
    image_exts = {'.jpg', '.jpeg', '.png', '.webp', '.tiff', '.bmp'}
    image_dirs = {}
    
    for root, dirs, files in os.walk(data_dir):
        img_files = [f for f in files if Path(f).suffix.lower() in image_exts]
        if img_files:
            rel_dir = os.path.relpath(root, data_dir)
            image_dirs[rel_dir] = img_files
    
    for dir_name, img_files in image_dirs.items():
        print(f"\n  === {dir_name}/ ===")
        print(f"  Total images: {len(img_files)}")
        
        # Extension distribution
        ext_counts = Counter(Path(f).suffix.lower() for f in img_files)
        print(f"  Extensions: {dict(ext_counts)}")
        
        # Sample image analysis
        sample_sizes = []
        sample_modes = []
        sample_file_sizes = []
        
        sample_files = img_files[:min(100, len(img_files))]
        for img_file in sample_files:
            img_path = os.path.join(data_dir, dir_name, img_file)
            try:
                img = Image.open(img_path)
                sample_sizes.append(img.size)
                sample_modes.append(img.mode)
                sample_file_sizes.append(os.path.getsize(img_path))
            except Exception as e:
                print(f"  WARNING: Could not read {img_file}: {e}")
        
        if sample_sizes:
            widths = [s[0] for s in sample_sizes]
            heights = [s[1] for s in sample_sizes]
            
            print(f"  Image modes: {Counter(sample_modes)}")
            print(f"  Width  - min: {min(widths)}, max: {max(widths)}, mean: {np.mean(widths):.0f}")
            print(f"  Height - min: {min(heights)}, max: {max(heights)}, mean: {np.mean(heights):.0f}")
            print(f"  Aspect ratios: {np.mean([w/h for w,h in sample_sizes]):.2f} mean")
            print(f"  File size - min: {min(sample_file_sizes)/1024:.0f}KB, max: {max(sample_file_sizes)/1024:.0f}KB, mean: {np.mean(sample_file_sizes)/1024:.0f}KB")
            
            # Size distribution
            unique_sizes = Counter(sample_sizes)
            if len(unique_sizes) <= 10:
                print(f"  Unique sizes: {dict(unique_sizes)}")
            else:
                print(f"  Unique sizes: {len(unique_sizes)} different sizes in sample")
    
    # ========== 4. Check Sample Submission Format ==========
    print(f"\n4. SUBMISSION FORMAT")
    print("-" * 50)
    for name in ['sample_submission.csv', 'sampleSubmission.csv', 'sample_sub.csv']:
        if name in csv_data:
            df = csv_data[name]
            print(f"  File: {name}")
            print(f"  Columns: {list(df.columns)}")
            print(f"  Rows: {len(df)}")
            print(f"  Sample:")
            print(df.head().to_string(index=False))
            break
    
    # ========== 5. Data Quality Checks ==========
    print(f"\n5. DATA QUALITY CHECKS")
    print("-" * 50)
    
    # Check if train images match train CSV
    for csv_name, df in csv_data.items():
        if 'train' in csv_name.lower():
            id_col = df.columns[0]  # Usually first column is ID
            for dir_name, img_files in image_dirs.items():
                if 'train' in dir_name.lower():
                    img_ids = set(Path(f).stem for f in img_files)
                    csv_ids = set(str(x) for x in df[id_col])
                    
                    matched = img_ids & csv_ids
                    only_csv = csv_ids - img_ids
                    only_img = img_ids - csv_ids
                    
                    print(f"  Train CSV ({csv_name}) vs Images ({dir_name}):")
                    print(f"    Matched: {len(matched)}")
                    print(f"    Only in CSV: {len(only_csv)}")
                    print(f"    Only in images: {len(only_img)}")
                    
                    if only_csv and len(only_csv) <= 5:
                        print(f"    Missing images: {only_csv}")
                    if only_img and len(only_img) <= 5:
                        print(f"    Extra images: {only_img}")
    
    # ========== 6. Generate Summary ==========
    summary = {
        'data_dir': data_dir,
        'total_files': len(all_files),
        'csv_files': csv_files,
        'image_directories': {k: len(v) for k, v in image_dirs.items()},
    }
    
    summary_path = os.path.join(data_dir, 'eda_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to {summary_path}")
    
    print(f"\n{'='*70}")
    print(f"  EDA COMPLETE")
    print(f"{'='*70}")
    
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, 
                        default=os.path.expanduser('~/freuid_challenge/data'))
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"ERROR: Data directory not found: {args.data_dir}")
        print("Run setup_server.sh first to download the data.")
        sys.exit(1)
    
    analyze_data(args.data_dir)


if __name__ == '__main__':
    main()
