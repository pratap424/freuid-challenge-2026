"""
FREUID Challenge 2026 - Data Augmentation Pipeline
Compatible with albumentations v2.x API.
All parameters use v2 naming: *_range instead of *_limit.
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np


def get_train_transforms(img_size=448, level='heavy'):
    """
    Get training augmentation pipeline.
    All augmentations verified against albumentations v2.x API.
    """
    if level == 'light':
        return A.Compose([
            A.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    elif level == 'medium':
        return A.Compose([
            A.RandomResizedCrop(size=(img_size, img_size), scale=(0.7, 1.0), ratio=(0.85, 1.15)),
            A.HorizontalFlip(p=0.5),
            A.Affine(
                translate_percent=(-0.05, 0.05),
                scale=(0.9, 1.1),
                rotate=(-10, 10),
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.5,
            ),
            A.OneOf([
                A.GaussNoise(std_range=(0.01, 0.05), p=1.0),
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.MedianBlur(blur_limit=(3, 3), p=1.0),
            ], p=0.3),
            A.ColorJitter(brightness=(0.85, 1.15), contrast=(0.85, 1.15),
                          saturation=(0.9, 1.1), hue=(-0.03, 0.03), p=0.4),
            A.ImageCompression(quality_range=(60, 100), p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    else:  # heavy — CRITICAL for generalization to test set
        return A.Compose([
            A.RandomResizedCrop(size=(img_size, img_size), scale=(0.6, 1.0), ratio=(0.8, 1.2)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            
            # Geometric distortions (simulate different capture angles/devices)
            A.Affine(
                translate_percent=(-0.08, 0.08),
                scale=(0.85, 1.15),
                rotate=(-15, 15),
                shear=(-10, 10),
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.6,
            ),
            A.Perspective(scale=(0.02, 0.06), p=0.3),
            
            # Color / Intensity (simulate different scanners/cameras)
            A.OneOf([
                A.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2),
                              saturation=(0.8, 1.2), hue=(-0.05, 0.05), p=1.0),
                A.RandomBrightnessContrast(brightness_limit=(-0.25, 0.25),
                                           contrast_limit=(-0.25, 0.25), p=1.0),
                A.HueSaturationValue(hue_shift_limit=(-10, 10),
                                     sat_shift_limit=(-20, 20),
                                     val_shift_limit=(-20, 20), p=1.0),
            ], p=0.6),
            
            # Noise & Blur (CRITICAL: simulate different capture devices)
            A.OneOf([
                A.GaussNoise(std_range=(0.01, 0.08), p=1.0),
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MedianBlur(blur_limit=(3, 5), p=1.0),
                A.MotionBlur(blur_limit=(3, 7), p=1.0),
            ], p=0.4),
            
            # Quality degradation (CRITICAL: simulate print-and-capture)
            A.OneOf([
                A.ImageCompression(quality_range=(30, 95), p=1.0),
                A.Downscale(scale_range=(0.4, 0.9), p=1.0),
            ], p=0.4),
            
            # Document-specific lighting conditions
            A.OneOf([
                A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_limit=(1, 3),
                               shadow_dimension=5, p=1.0),
                A.RandomToneCurve(scale=0.15, p=1.0),
                A.CLAHE(clip_limit=(1, 4), p=1.0),
            ], p=0.25),
            
            # Channel operations (force color invariance)
            A.OneOf([
                A.ChannelShuffle(p=1.0),
                A.ToGray(p=1.0),
                A.ChannelDropout(p=1.0),
            ], p=0.05),
            
            # Cutout / CoarseDropout (v2 API)
            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(0.02, 0.08),
                hole_width_range=(0.02, 0.08),
                fill=0,
                p=0.3,
            ),
            
            # Normalize
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])


def get_valid_transforms(img_size=448):
    """Validation / test transforms."""
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_tta_transforms(img_size=448):
    """Test-Time Augmentation transforms."""
    transforms = [
        # 0: Original
        A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        # 1: Horizontal flip
        A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        # 2: Slight scale up
        A.Compose([
            A.Resize(height=int(img_size*1.1), width=int(img_size*1.1)),
            A.CenterCrop(height=img_size, width=img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        # 3: Slight scale down + pad
        A.Compose([
            A.Resize(height=int(img_size*0.9), width=int(img_size*0.9)),
            A.PadIfNeeded(min_height=img_size, min_width=img_size,
                         border_mode=cv2.BORDER_REFLECT_101),
            A.CenterCrop(height=img_size, width=img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        # 4: Brightness up
        A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.RandomBrightnessContrast(brightness_limit=(0.1, 0.1), contrast_limit=(0, 0), p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        # 5: Brightness down
        A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.RandomBrightnessContrast(brightness_limit=(-0.1, -0.1), contrast_limit=(0, 0), p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        # 6: JPEG compression
        A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.ImageCompression(quality_range=(60, 60), p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        # 7: Slight noise
        A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.GaussNoise(std_range=(0.02, 0.02), p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
    ]
    
    return transforms


# ==================== Mixup / CutMix ====================

def mixup(images, labels, alpha=0.4):
    """Apply Mixup augmentation."""
    if alpha <= 0:
        return images, labels, labels, 1.0
    
    lam = np.random.beta(alpha, alpha)
    batch_size = images.size(0)
    index = np.random.permutation(batch_size)
    
    import torch
    index = torch.tensor(index, device=images.device)
    
    mixed_images = lam * images + (1 - lam) * images[index]
    labels_a, labels_b = labels, labels[index]
    
    return mixed_images, labels_a, labels_b, lam


def cutmix(images, labels, alpha=1.0):
    """Apply CutMix augmentation."""
    if alpha <= 0:
        return images, labels, labels, 1.0
    
    import torch
    
    lam = np.random.beta(alpha, alpha)
    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)
    
    _, _, H, W = images.shape
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)
    
    cy = np.random.randint(H)
    cx = np.random.randint(W)
    
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    
    images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
    lam = 1 - (y2 - y1) * (x2 - x1) / (H * W)
    
    labels_a, labels_b = labels, labels[index]
    
    return images, labels_a, labels_b, lam


def mix_criterion(criterion, pred, labels_a, labels_b, lam):
    """Compute loss with mixup/cutmix interpolation."""
    return lam * criterion(pred, labels_a) + (1 - lam) * criterion(pred, labels_b)
