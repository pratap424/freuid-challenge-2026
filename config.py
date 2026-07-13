"""
FREUID Challenge 2026 - Configuration
All hyperparameters and paths in one place.
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:
    # ======================== PATHS ========================
    data_dir: str = os.path.expanduser("~/freuid_challenge/data")
    output_dir: str = os.path.expanduser("~/freuid_challenge/outputs")
    checkpoint_dir: str = os.path.expanduser("~/freuid_challenge/checkpoints")
    
    # ======================== DATA ========================
    train_csv: str = "train_labels.csv"
    test_csv: str = "sample_submission.csv"  # test IDs come from here
    sample_sub_csv: str = "sample_submission.csv"
    train_images_dir: str = "train/train"         # double-nested!
    test_images_dir: str = "public_test/public_test"  # double-nested!
    image_col: str = "id"
    label_col: str = "label"
    # Extra metadata columns (training only, not available at test time)
    digital_col: str = "is_digital"
    type_col: str = "type"
    
    # ======================== MODEL ========================
    model_name: str = "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k"
    num_classes: int = 1  # binary: single sigmoid output
    pretrained: bool = True
    drop_rate: float = 0.3
    drop_path_rate: float = 0.2
    
    # ======================== TRAINING ========================
    img_size: int = 384
    batch_size: int = 16
    num_workers: int = 8
    epochs: int = 15
    
    # Optimizer
    optimizer: str = "adamw"
    lr: float = 1e-4
    backbone_lr_mult: float = 1.0  # full fine-tuning — backbone MUST learn fraud features
    weight_decay: float = 0.05
    
    # Scheduler
    scheduler: str = "cosine_warmup"
    warmup_epochs: int = 2
    min_lr: float = 1e-7
    
    # Training tricks
    use_amp: bool = True
    gradient_accumulation: int = 1
    max_grad_norm: float = 1.0
    ema_decay: float = 0.9998
    label_smoothing: float = 0.1
    
    # Loss
    loss_fn: str = "bce_focal"  # 'bce', 'focal', 'bce_focal'
    focal_gamma: float = 2.0
    focal_alpha: float = 0.75
    pos_weight: float = 1.0  # adjust based on class imbalance
    
    # ======================== VALIDATION ========================
    n_folds: int = 5
    fold: int = 0
    val_interval: int = 1
    
    # ======================== AUGMENTATION ========================
    aug_level: str = "heavy"  # 'light', 'medium', 'heavy'
    use_ela: bool = False  # start without ELA, add in later phase
    use_freq: bool = False  # enable in phase 3
    mixup_alpha: float = 0.4
    cutmix_alpha: float = 1.0
    mix_prob: float = 0.5
    
    # ======================== TTA ========================
    use_tta: bool = True
    tta_transforms: int = 8
    
    # ======================== ENSEMBLE ========================
    ensemble_models: List[str] = field(default_factory=lambda: [
        "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",
        "convnextv2_large.fcmae_ft_in22k_in1k_384",
        "swin_large_patch4_window12_384.ms_in22k_ft_in1k",
        "tf_efficientnetv2_l.in21k_ft_in1k",
        "vit_large_patch14_clip_336.openai_ft_in12k_in1k",
        "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",  # different seed
        "convnextv2_base.fcmae_ft_in22k_in1k_384",
        "vit_large_patch14_dinov2.lvd142m",
    ])
    
    # ======================== MISC ========================
    seed: int = 42
    debug: bool = False
    wandb_project: str = "freuid-challenge-2026"
    experiment_name: str = "baseline"
    
    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    @property
    def train_images_path(self):
        return os.path.join(self.data_dir, self.train_images_dir)
    
    @property
    def test_images_path(self):
        return os.path.join(self.data_dir, self.test_images_dir)
    
    @property
    def train_csv_path(self):
        return os.path.join(self.data_dir, self.train_csv)
    
    @property
    def test_csv_path(self):
        return os.path.join(self.data_dir, self.test_csv)
    
    @property  
    def sample_sub_path(self):
        return os.path.join(self.data_dir, self.sample_sub_csv)


# ======================== PRESET CONFIGS ========================
# Tuned for 24GB GPU. Start with small models, scale up once validated.

def get_effnet_b3_config(fold=0):
    """RECOMMENDED FIRST MODEL: 12M params, fast, won't overfit easily."""
    return Config(
        model_name="efficientnet_b3.ra2_in1k",
        img_size=300, batch_size=32, gradient_accumulation=1,
        lr=1e-4, backbone_lr_mult=1.0,
        drop_rate=0.3, drop_path_rate=0.2,
        epochs=15, warmup_epochs=2,
        weight_decay=0.05, label_smoothing=0.1,
        fold=fold, experiment_name=f"effnet_b3_f{fold}",
    )

def get_convnext_small_config(fold=0):
    """Good balance of capacity and generalization. 50M params."""
    return Config(
        model_name="convnext_small.fb_in22k_ft_in1k_384",
        img_size=384, batch_size=16, gradient_accumulation=1,
        lr=5e-5, backbone_lr_mult=1.0,
        drop_rate=0.3, drop_path_rate=0.3,
        epochs=15, warmup_epochs=2,
        weight_decay=0.05, label_smoothing=0.1,
        fold=fold, experiment_name=f"convnext_small_f{fold}",
    )

def get_eva02_config(fold=0):
    """Large model — use ONLY after smaller models show AUC > 0.7."""
    return Config(
        model_name="eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",
        img_size=448, batch_size=4, gradient_accumulation=4,
        lr=2e-5, backbone_lr_mult=1.0,  # FULL fine-tuning
        drop_rate=0.4, drop_path_rate=0.3,
        epochs=10, warmup_epochs=2,
        weight_decay=0.05, label_smoothing=0.1,
        fold=fold, experiment_name=f"eva02_large_f{fold}",
    )

def get_convnextv2_config(fold=0):
    return Config(
        model_name="convnextv2_large.fcmae_ft_in22k_in1k_384",
        img_size=384, batch_size=8, gradient_accumulation=2,
        lr=3e-5, backbone_lr_mult=1.0,
        drop_rate=0.4, drop_path_rate=0.3,
        epochs=12, warmup_epochs=2,
        weight_decay=0.05, label_smoothing=0.1,
        fold=fold, experiment_name=f"convnextv2_large_f{fold}",
    )

def get_swin_config(fold=0):
    return Config(
        model_name="swin_large_patch4_window12_384.ms_in22k_ft_in1k",
        img_size=384, batch_size=8, gradient_accumulation=2,
        lr=3e-5, backbone_lr_mult=1.0,
        drop_rate=0.4, drop_path_rate=0.3,
        epochs=12, warmup_epochs=2,
        weight_decay=0.05, label_smoothing=0.1,
        fold=fold, experiment_name=f"swin_large_f{fold}",
    )

def get_effnetv2_config(fold=0):
    return Config(
        model_name="tf_efficientnetv2_l.in21k_ft_in1k",
        img_size=384, batch_size=8, gradient_accumulation=2,
        lr=3e-5, backbone_lr_mult=1.0,
        epochs=12, warmup_epochs=2,
        weight_decay=0.05, label_smoothing=0.1,
        fold=fold, experiment_name=f"effnetv2_l_f{fold}",
    )

def get_vit_clip_config(fold=0):
    return Config(
        model_name="vit_large_patch14_clip_336.openai_ft_in12k_in1k",
        img_size=336, batch_size=8, gradient_accumulation=2,
        lr=2e-5, backbone_lr_mult=1.0,
        epochs=12, warmup_epochs=2,
        weight_decay=0.05, label_smoothing=0.1,
        fold=fold, experiment_name=f"vit_clip_large_f{fold}",
    )

def get_dinov2_config(fold=0):
    return Config(
        model_name="vit_large_patch14_dinov2.lvd142m",
        img_size=518, batch_size=2, gradient_accumulation=8,
        lr=1e-5, backbone_lr_mult=1.0,
        epochs=10, warmup_epochs=2,
        weight_decay=0.05, label_smoothing=0.1,
        fold=fold, experiment_name=f"dinov2_large_f{fold}",
    )

MODEL_CONFIGS = {
    'effnet_b3': get_effnet_b3_config,      # START HERE
    'convnext_small': get_convnext_small_config,  # THEN HERE
    'eva02': get_eva02_config,
    'convnextv2': get_convnextv2_config,
    'swin': get_swin_config,
    'effnetv2': get_effnetv2_config,
    'vit_clip': get_vit_clip_config,
    'dinov2': get_dinov2_config,
}
