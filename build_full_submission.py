"""Build the full-competition submission (public + private test predictions).

Blends raw per-checkpoint predictions saved by ensemble_inference.py:
  outputs/predictions/raw/{model}_f{fold}[_tta].npy       (public test)
  outputs/predictions/raw/priv_{model}_f{fold}[_tta].npy  (private test)

Blend formula: alpha * mean + (1 - alpha) * max  (alpha=0.5 was best on public LB).
For each split, prefers _tta raws when available, falls back to non-TTA.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

RAW_DIR = os.path.join('outputs', 'predictions', 'raw')


def load_split(prefix):
    """Load raw prediction arrays for one split ('' = public, 'priv_' = private)."""
    tta = sorted(glob.glob(os.path.join(RAW_DIR, f'{prefix}*_f*_tta.npy')))
    plain = sorted(glob.glob(os.path.join(RAW_DIR, f'{prefix}*_f*.npy')))
    plain = [p for p in plain if not p.endswith('_tta.npy')]
    files = tta if tta else plain
    if prefix == '':
        files = [f for f in files if not os.path.basename(f).startswith('priv_')]
    if not files:
        return None, None
    preds = [np.load(f) for f in files]
    ids = pd.read_csv(os.path.join(RAW_DIR, f'{prefix}test_ids.csv'))['id']
    for f, p in zip(files, preds):
        assert len(p) == len(ids), f'{f}: {len(p)} preds vs {len(ids)} ids'
    print(f'  split={prefix or "public"}: {len(files)} checkpoint preds x {len(ids)} images')
    for f in files:
        print(f'    {os.path.basename(f)}')
    return np.array(preds), ids


def blend(preds, alpha):
    return alpha * preds.mean(axis=0) + (1 - alpha) * preds.max(axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--alpha', type=float, default=0.5)
    ap.add_argument('--sample_sub', type=str,
                    default=os.path.expanduser('~/freuid_challenge/data/sample_submission.csv'))
    ap.add_argument('--out', type=str, default=None)
    args = ap.parse_args()

    parts = []
    for prefix in ['', 'priv_']:
        preds, ids = load_split(prefix)
        if preds is None:
            print(f'  WARNING: no raw preds for split "{prefix or "public"}"')
            continue
        parts.append(pd.DataFrame({'id': ids, 'label': blend(preds, args.alpha)}))

    pred_df = pd.concat(parts, ignore_index=True)
    sub = pd.read_csv(args.sample_sub)[['id']].merge(pred_df, on='id', how='left')
    n_missing = sub['label'].isna().sum()
    sub['label'] = sub['label'].fillna(0.5)
    print(f'rows={len(sub)} predicted={len(sub) - n_missing} missing(0.5)={n_missing}')

    out = args.out or os.path.join('outputs', 'predictions',
                                   f'submission_FULL_a{int(args.alpha * 100)}.csv')
    sub.to_csv(out, index=False)
    print(f'saved {out}')


if __name__ == '__main__':
    main()
