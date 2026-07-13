"""
FREUID Challenge 2026 - Model Definitions
Flexible model wrapper using timm library.
Supports all architectures in our ensemble strategy.
"""
import torch
import torch.nn as nn
import timm
from copy import deepcopy


class FREUIDModel(nn.Module):
    """
    Generic model wrapper for fraud detection.
    Uses timm backbone with custom classification head.
    """
    
    def __init__(
        self,
        model_name: str = "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",
        pretrained: bool = True,
        num_classes: int = 1,
        in_channels: int = 3,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.2,
        use_gem_pool: bool = True,
    ):
        super().__init__()
        
        self.model_name = model_name
        
        # Create backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # remove classifier
            in_chans=in_channels,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )
        
        # Get feature dimension
        self.feat_dim = self.backbone.num_features
        
        # Optional GeM pooling (if backbone uses global pooling)
        self.use_gem = use_gem_pool
        if use_gem_pool:
            self.gem = GeM(p=3.0)
        
        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Dropout(p=0.3),
            nn.Linear(self.feat_dim, 512),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes),
        )
        
        # Initialize head
        self._init_head()
    
    def _init_head(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        features = self.backbone(x)
        
        # If backbone returns spatial features, apply global pooling
        if len(features.shape) == 4:  # (B, C, H, W)
            if self.use_gem:
                features = self.gem(features)
            else:
                features = features.mean(dim=[2, 3])
        elif len(features.shape) == 3:  # (B, N, C) from ViT
            features = features.mean(dim=1)  # average patch tokens
        
        logits = self.head(features)
        return logits.squeeze(-1)
    
    def get_backbone_params(self):
        return self.backbone.parameters()
    
    def get_head_params(self):
        params = list(self.head.parameters())
        if self.use_gem:
            params += list(self.gem.parameters())
        return params


class GeM(nn.Module):
    """Generalized Mean Pooling."""
    
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
    
    def forward(self, x):
        return x.clamp(min=self.eps).pow(self.p).mean(dim=[2, 3]).pow(1.0 / self.p)


class ModelEMA:
    """
    Exponential Moving Average of model parameters.
    Critical for stable performance and ~0.5-1% metric improvement.
    """
    
    def __init__(self, model, decay=0.9998):
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
    
    @torch.no_grad()
    def update(self, model):
        for ema_p, model_p in zip(self.module.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1.0 - self.decay)
    
    def state_dict(self):
        return self.module.state_dict()
    
    def load_state_dict(self, state_dict):
        self.module.load_state_dict(state_dict)


# ==================== Loss Functions ====================

class FocalLoss(nn.Module):
    """Binary Focal Loss for handling class imbalance."""
    
    def __init__(self, gamma=2.0, alpha=0.75, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        probs = torch.sigmoid(logits)
        
        # pt = p if y=1, else 1-p
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        
        # Alpha weighting
        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        loss = alpha_weight * focal_weight * bce
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class CompositeLoss(nn.Module):
    """
    Combined loss for FREUID metric optimization.
    BCE + Focal + optional AUC proxy loss.
    """
    
    def __init__(
        self,
        bce_weight: float = 0.5,
        focal_weight: float = 0.5,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.75,
        pos_weight: float = 1.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.label_smoothing = label_smoothing
        
        pw = torch.tensor([pos_weight])
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pw)
        self.focal = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
    
    def forward(self, logits, targets):
        # Apply label smoothing
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        # Move pos_weight to correct device
        self.bce.pos_weight = self.bce.pos_weight.to(logits.device)
        
        loss = 0
        if self.bce_weight > 0:
            loss += self.bce_weight * self.bce(logits, targets)
        if self.focal_weight > 0:
            loss += self.focal_weight * self.focal(logits, targets)
        
        return loss


def build_model(config):
    """Build model from config."""
    in_channels = 3
    if config.use_ela:
        in_channels += 1
    
    model = FREUIDModel(
        model_name=config.model_name,
        pretrained=config.pretrained,
        num_classes=config.num_classes,
        in_channels=in_channels,
        drop_rate=config.drop_rate,
        drop_path_rate=config.drop_path_rate,
    )
    
    return model


def build_loss(config):
    """Build loss function from config."""
    return CompositeLoss(
        bce_weight=0.5,
        focal_weight=0.5,
        focal_gamma=config.focal_gamma,
        focal_alpha=config.focal_alpha,
        pos_weight=config.pos_weight,
        label_smoothing=config.label_smoothing,
    )


def count_parameters(model):
    """Count trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
