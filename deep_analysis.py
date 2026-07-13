"""
FREUID Challenge 2026 - Deep Data Analysis
Run on server BEFORE training to understand where fraud is hard.

Usage: python deep_analysis.py --data_dir ~/freuid_challenge/data
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from PIL import Image


def deep_analysis(data_dir):
    csv_path = os.path.join(data_dir, 'train_labels.csv')
    df = pd.read_csv(csv_path)
    
    print(f"\n{'='*70}")
    print(f"  DEEP DATA ANALYSIS — FREUID Challenge 2026")
    print(f"{'='*70}")
    print(f"\nTotal training samples: {len(df)}")
    
    # ===== 1. Type distribution =====
    print(f"\n{'='*70}")
    print(f"  1. DOCUMENT TYPE × LABEL")
    print(f"{'='*70}")
    for doc_type in sorted(df['type'].unique()):
        subset = df[df['type'] == doc_type]
        n_bona = (subset['label'] == 0).sum()
        n_fraud = (subset['label'] == 1).sum()
        pct_fraud = n_fraud / len(subset) * 100
        bar = '█' * int(pct_fraud / 2) + '░' * (50 - int(pct_fraud / 2))
        print(f"  {doc_type:20s} | {len(subset):>6d} total | {n_bona:>5d} bona {n_fraud:>5d} fraud | {pct_fraud:5.1f}% fraud | {bar}")
    
    # ===== 2. is_digital distribution =====
    print(f"\n{'='*70}")
    print(f"  2. CAPTURE MODE × LABEL")
    print(f"{'='*70}")
    for is_dig in [True, False]:
        subset = df[df['is_digital'] == is_dig]
        n_bona = (subset['label'] == 0).sum()
        n_fraud = (subset['label'] == 1).sum()
        pct_fraud = n_fraud / len(subset) * 100
        mode = "DIGITAL" if is_dig else "PHYSICAL"
        print(f"  {mode:20s} | {len(subset):>6d} total | {n_bona:>5d} bona {n_fraud:>5d} fraud | {pct_fraud:5.1f}% fraud")
    
    # ===== 3. Full cross-tabulation =====
    print(f"\n{'='*70}")
    print(f"  3. FULL CROSS-TAB: type × is_digital × label")
    print(f"{'='*70}")
    print(f"  {'Type':20s} {'Mode':8s} | {'Total':>6s} {'Bona':>6s} {'Fraud':>6s} {'%Fraud':>7s}")
    print(f"  {'-'*20} {'-'*8} | {'-'*6} {'-'*6} {'-'*6} {'-'*7}")
    
    for doc_type in sorted(df['type'].unique()):
        for is_dig in [True, False]:
            subset = df[(df['type'] == doc_type) & (df['is_digital'] == is_dig)]
            if len(subset) == 0:
                continue
            n_bona = (subset['label'] == 0).sum()
            n_fraud = (subset['label'] == 1).sum()
            pct = n_fraud / len(subset) * 100
            mode = "DIG" if is_dig else "PHY"
            print(f"  {doc_type:20s} {mode:8s} | {len(subset):>6d} {n_bona:>6d} {n_fraud:>6d} {pct:>6.1f}%")
    
    # ===== 4. Image size analysis per type =====
    print(f"\n{'='*70}")
    print(f"  4. IMAGE SIZE PER TYPE (sample of 20 per type)")
    print(f"{'='*70}")
    
    for doc_type in sorted(df['type'].unique()):
        subset = df[df['type'] == doc_type].head(20)
        sizes = []
        for _, row in subset.iterrows():
            img_path = os.path.join(data_dir, row['image_path'])
            # Handle double-nested
            if not os.path.exists(img_path):
                parts = row['image_path'].split('/')
                img_path = os.path.join(data_dir, parts[0], parts[0], parts[1])
            try:
                img = Image.open(img_path)
                sizes.append(img.size)
            except:
                pass
        
        if sizes:
            from collections import Counter
            size_counts = Counter(sizes)
            print(f"\n  {doc_type}:")
            for sz, cnt in size_counts.most_common():
                print(f"    {sz[0]}×{sz[1]}: {cnt}")
    
    # ===== 5. File size analysis: does it correlate with fraud? =====
    print(f"\n{'='*70}")
    print(f"  5. FILE SIZE vs FRAUD (sample of 500)")
    print(f"{'='*70}")
    
    sample = df.sample(min(500, len(df)), random_state=42)
    bona_sizes = []
    fraud_sizes = []
    
    for _, row in sample.iterrows():
        img_path = os.path.join(data_dir, row['image_path'])
        if not os.path.exists(img_path):
            parts = row['image_path'].split('/')
            img_path = os.path.join(data_dir, parts[0], parts[0], parts[1])
        try:
            fsize = os.path.getsize(img_path) / 1024  # KB
            if row['label'] == 0:
                bona_sizes.append(fsize)
            else:
                fraud_sizes.append(fsize)
        except:
            pass
    
    if bona_sizes and fraud_sizes:
        print(f"  Bonafide file sizes: mean={np.mean(bona_sizes):.0f}KB, std={np.std(bona_sizes):.0f}KB, median={np.median(bona_sizes):.0f}KB")
        print(f"  Fraud    file sizes: mean={np.mean(fraud_sizes):.0f}KB, std={np.std(fraud_sizes):.0f}KB, median={np.median(fraud_sizes):.0f}KB")
        diff = abs(np.mean(bona_sizes) - np.mean(fraud_sizes))
        print(f"  Difference: {diff:.0f}KB — {'SIGNIFICANT' if diff > 10 else 'negligible'}")
    
    # ===== 6. Strata count for CV =====
    print(f"\n{'='*70}")
    print(f"  6. STRATIFICATION STRATA FOR CV")
    print(f"{'='*70}")
    strat_key = df['label'].astype(str) + '_' + df['type'] + '_' + df['is_digital'].astype(str)
    print(f"  Compound strata (label × type × is_digital): {strat_key.nunique()}")
    print(f"  Smallest stratum: {strat_key.value_counts().min()} samples")
    print(f"  Largest stratum:  {strat_key.value_counts().max()} samples")
    print(f"\n  All strata:")
    for stratum, count in strat_key.value_counts().sort_index().items():
        print(f"    {stratum:40s}: {count:>5d}")
    
    print(f"\n{'='*70}")
    print(f"  ANALYSIS COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=os.path.expanduser('~/freuid_challenge/data'))
    args = parser.parse_args()
    deep_analysis(args.data_dir)
