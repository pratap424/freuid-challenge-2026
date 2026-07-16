"""FREUID 2026 Docker sandbox entrypoint (contract-compliant).

Contract:
  /data/         (ro)  flat dir of images (.jpeg/.jpg/.png/.webp/.bmp/.tif/.tiff)
  /submissions/  (rw)  writes submission.csv with columns id,label
  /weights/      (ro)  model checkpoints (mounted; no runtime downloads)
  --network none        all weights loaded locally, HF_HUB_OFFLINE enforced

Output: one row per image, id = filename without extension,
label = finite float fraud score (higher = more fraud).
"""
import os

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

import glob

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import cv2

from config import MODEL_CONFIGS
from models import build_model
from augmentations import get_valid_transforms, get_tta_transforms

DATA_DIR = os.environ.get('DATA_DIR', '/data')
OUT_DIR = os.environ.get('OUT_DIR', '/submissions')
WEIGHTS_DIR = os.environ.get('WEIGHTS_DIR', '/weights')
TTA_VIEWS = int(os.environ.get('TTA_VIEWS', '2'))
ALPHA = float(os.environ.get('BLEND_ALPHA', '0.5'))
EXTS = {'.jpeg', '.jpg', '.png', '.webp', '.bmp', '.tif', '.tiff'}

ENSEMBLE = [
    ('convnext_small', [0, 1, 2, 3, 4], 'convnext_small_f{f}.pth', 64),
    ('eva02',          [0],             'eva02_large_f{f}.pth',     16),
]


class FlatImageDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx], cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((384, 384, 3), dtype=np.uint8)  # unreadable -> neutral input
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.transform(image=img)['image']


def load_checkpoint(model, path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    state = ckpt.get('ema_state_dict') or ckpt.get('model_state_dict') or ckpt
    model.load_state_dict(state)
    return model


@torch.no_grad()
def predict(model, paths, transform, batch_size):
    ds = FlatImageDataset(paths, transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=int(os.environ.get('NUM_WORKERS', '4')),
                        pin_memory=torch.cuda.is_available())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    preds = []
    for images in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device, enabled=(device == 'cuda')):
            logits = model(images)
        preds.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(preds)


def main():
    paths = sorted(p for p in glob.glob(os.path.join(DATA_DIR, '*'))
                   if os.path.splitext(p)[1].lower() in EXTS)
    ids = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    assert len(ids) == len(set(ids)), 'duplicate image ids in /data'
    print(f'images: {len(paths)}  tta_views: {TTA_VIEWS}  alpha: {ALPHA}', flush=True)

    all_preds = []
    for name, folds, pattern, bs in ENSEMBLE:
        cfg = MODEL_CONFIGS[name](0)
        cfg.pretrained = False
        transforms = (get_tta_transforms(cfg.img_size)[:TTA_VIEWS]
                      if TTA_VIEWS > 1 else [get_valid_transforms(cfg.img_size)])
        for f in folds:
            wpath = os.path.join(WEIGHTS_DIR, pattern.format(f=f))
            if not os.path.exists(wpath):
                print(f'WARNING: missing {wpath}, skipping', flush=True)
                continue
            model = load_checkpoint(build_model(cfg), wpath)
            model = model.cuda().eval() if torch.cuda.is_available() else model.eval()
            view_preds = [predict(model, paths, t, bs) for t in transforms]
            all_preds.append(np.mean(view_preds, axis=0))
            print(f'{name} f{f}: done ({len(transforms)} views)', flush=True)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    assert all_preds, 'no checkpoints found in /weights'
    arr = np.array(all_preds)
    final = ALPHA * arr.mean(axis=0) + (1 - ALPHA) * arr.max(axis=0)

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, 'submission.csv')
    pd.DataFrame({'id': ids, 'label': final}).to_csv(out, index=False)
    print(f'wrote {out} ({len(ids)} rows)', flush=True)


if __name__ == '__main__':
    main()
