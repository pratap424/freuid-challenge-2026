"""
FREUID Challenge 2026 - Forensic Feature Extraction
ELA (Error Level Analysis) and noise residual features
for document fraud detection.
"""
import cv2
import numpy as np
from io import BytesIO
from PIL import Image


def compute_ela(image_rgb, quality=90, scale=15.0):
    """
    Compute Error Level Analysis (ELA).
    
    Principle: Save image at a known JPEG quality, then compute the
    pixel-level difference with the original. Regions that were
    previously compressed at a different quality (e.g., pasted from
    another image) will show higher error levels.
    
    Args:
        image_rgb: numpy array (H, W, 3) in RGB, uint8
        quality: JPEG compression quality for re-save
        scale: amplification factor for the difference
    
    Returns:
        ela: numpy array (H, W) single-channel ELA map, float32, [0,1]
    """
    # Convert to PIL for JPEG re-compression
    pil_img = Image.fromarray(image_rgb)
    
    # Re-compress at known quality
    buffer = BytesIO()
    pil_img.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    recompressed = np.array(Image.open(buffer))
    
    # Compute absolute difference
    diff = np.abs(image_rgb.astype(np.float32) - recompressed.astype(np.float32))
    
    # Average across channels and scale
    ela = diff.mean(axis=2) * scale
    
    # Clip to [0, 255] then normalize to [0, 1]
    ela = np.clip(ela, 0, 255) / 255.0
    
    return ela.astype(np.float32)


def compute_noise_residual(image_rgb, ksize=3):
    """
    Compute noise residual using median filter denoising.
    
    Principle: Apply a median filter (which removes noise while preserving
    edges), then subtract from original. The residual reveals the noise
    pattern. Authentic images have uniform noise; fraud documents have
    inconsistent noise at edit boundaries.
    
    Args:
        image_rgb: numpy array (H, W, 3) in RGB, uint8
        ksize: kernel size for median filter
    
    Returns:
        residual: numpy array (H, W) single-channel noise residual, float32, [0,1]
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    
    # Apply median filter
    denoised = cv2.medianBlur(image_rgb, ksize)
    denoised_gray = cv2.cvtColor(denoised, cv2.COLOR_RGB2GRAY).astype(np.float32)
    
    # Compute residual
    residual = np.abs(gray - denoised_gray)
    
    # Normalize to [0, 1]
    if residual.max() > 0:
        residual = residual / residual.max()
    
    return residual.astype(np.float32)


# ==================== SRM Filters ====================

def get_srm_kernels():
    """
    Get the 3 basic SRM (Steganalysis Rich Model) high-pass filters.
    These capture local noise patterns that are disrupted by image manipulation.
    """
    # First-order edge filter
    srm1 = np.array([
        [0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0],
        [0,  1, -2,  1,  0],
        [0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0],
    ], dtype=np.float32)
    
    # Second-order edge filter (square)
    srm2 = np.array([
        [0,  0,  0,  0,  0],
        [0, -1,  2, -1,  0],
        [0,  2, -4,  2,  0],
        [0, -1,  2, -1,  0],
        [0,  0,  0,  0,  0],
    ], dtype=np.float32)
    
    # Third-order edge filter
    srm3 = np.array([
        [-1,  2, -2,  2, -1],
        [ 2, -6,  8, -6,  2],
        [-2,  8, -12, 8, -2],
        [ 2, -6,  8, -6,  2],
        [-1,  2, -2,  2, -1],
    ], dtype=np.float32)
    
    return [srm1, srm2, srm3]


def compute_srm_features(image_rgb):
    """
    Apply SRM high-pass filters to extract noise/manipulation features.
    
    Args:
        image_rgb: numpy array (H, W, 3) in RGB, uint8
    
    Returns:
        srm_features: numpy array (H, W, 3) three SRM filter responses, float32
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    kernels = get_srm_kernels()
    
    features = []
    for kernel in kernels:
        filtered = cv2.filter2D(gray, -1, kernel)
        # Normalize to [0, 1]
        filtered = np.abs(filtered)
        if filtered.max() > 0:
            filtered = filtered / filtered.max()
        features.append(filtered)
    
    return np.stack(features, axis=-1).astype(np.float32)


def compute_multi_scale_ela(image_rgb, qualities=[90, 75, 50]):
    """
    Compute ELA at multiple JPEG quality levels.
    Different fraud types are visible at different quality levels.
    
    Returns:
        multi_ela: numpy array (H, W, len(qualities)), float32
    """
    elas = []
    for q in qualities:
        ela = compute_ela(image_rgb, quality=q, scale=15.0)
        elas.append(ela)
    
    return np.stack(elas, axis=-1).astype(np.float32)


# ==================== Combined Feature Extraction ====================

def extract_forensic_features(image_rgb, mode='ela'):
    """
    Extract forensic features from an image.
    
    Args:
        image_rgb: numpy array (H, W, 3) in RGB, uint8
        mode: 'ela' for single ELA, 'multi_ela' for multi-scale ELA,
              'noise' for noise residual, 'srm' for SRM features,
              'all' for ELA + noise + SRM (7 channels total)
    
    Returns:
        features: numpy array (H, W, C), float32
    """
    if mode == 'ela':
        return compute_ela(image_rgb)[..., np.newaxis]
    
    elif mode == 'multi_ela':
        return compute_multi_scale_ela(image_rgb)
    
    elif mode == 'noise':
        return compute_noise_residual(image_rgb)[..., np.newaxis]
    
    elif mode == 'srm':
        return compute_srm_features(image_rgb)
    
    elif mode == 'all':
        ela = compute_ela(image_rgb)[..., np.newaxis]  # 1 channel
        noise = compute_noise_residual(image_rgb)[..., np.newaxis]  # 1 channel
        srm = compute_srm_features(image_rgb)  # 3 channels
        return np.concatenate([ela, noise, srm], axis=-1)  # 5 channels
    
    else:
        raise ValueError(f"Unknown mode: {mode}")
