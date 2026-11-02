"""
Loss functions - models_improved.py
-----------------------------------------------------------------
ImprovedLoss = FocalLoss(0.3) + DiceLoss(0.3) + LovászLoss(0.4)

Lovasz loss directly optimizes IoU; the reported mIoU gain over CE/Focal is 3-5% for classes with few samples.
The implementation is included and requires no additional installation.
Reference: Berman et al. 2018, "The Lovasz-Softmax Loss".
-----------------------------------------------------------------
"""

import glob, os
from typing import Iterable, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import config
from models import FocalLoss, DiceLoss


# =================================================================
# Lovasz loss implementation
# =================================================================
def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    p     = len(gt_sorted)
    gts   = gt_sorted.sum()
    inter = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jacc  = 1. - inter / union
    if p > 1:
        jacc[1:p] = jacc[1:p] - jacc[0:-1]
    return jacc


def _lovasz_softmax_flat(probs: torch.Tensor,
                          labels: torch.Tensor) -> torch.Tensor:
    """
    probs:  (P, C) - P valid pixels and C classes
    labels: (P,)   - ground-truth classes
    """
    C      = probs.size(1)
    losses = []

    for c in range(1, C):   # Skip ignore slot 0
        fg       = (labels == c).float()
        if fg.sum() == 0:
            continue
        errors   = (fg - probs[:, c]).abs()
        sort_idx = errors.sort(descending=True)[1]
        errors_s = errors[sort_idx]
        gt_s     = fg[sort_idx]
        grad     = _lovasz_grad(gt_s).detach()
        losses.append(torch.dot(errors_s, grad))

    return torch.stack(losses).mean() if losses else probs.sum() * 0.


class LovaszLoss(nn.Module):
    """Lovasz-Softmax loss for directly optimizing multiclass IoU."""
    def __init__(self, ignore_index: int = config.IGNORE_INDEX):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        probs = F.softmax(pred, dim=1)                  # (B,C,H,W)
        B, C, H, W = probs.shape

        probs_flat  = probs.permute(0, 2, 3, 1).contiguous().view(-1, C)
        target_flat = target.view(-1)

        valid       = (target_flat != self.ignore_index)
        probs_flat  = probs_flat[valid]
        target_flat = target_flat[valid]

        if probs_flat.numel() == 0:
            return pred.sum() * 0.

        return _lovasz_softmax_flat(probs_flat, target_flat)


# =================================================================
# Dynamic class weights
# =================================================================
def compute_dynamic_weights_from_paths(mask_paths: Iterable[str],
                                       num_classes: int,
                                       ignore_index: int,
                                       smoothing: float = 0.15) -> List[float]:
    files = [str(path) for path in mask_paths if path]
    if not files:
        print("[warning] No mask files provided, using default weights")
        return list(config.CLASS_WEIGHTS)

    counts = np.zeros(num_classes + 1, dtype=np.int64)
    for path in files:
        m = np.array(Image.open(path))
        for c in range(num_classes + 1):
            counts[c] += (m == c).sum()

    valid_counts = counts[1:num_classes + 1].astype(np.float64)
    total = valid_counts.sum()
    if total == 0:
        return [1.0] * num_classes

    freq = valid_counts / total
    weights = (1.0 - smoothing) / np.maximum(freq, 1e-8) + smoothing
    weights = (weights / weights.mean()).tolist()

    print(f"[dynamic-weights] Computed from {len(files)} masks")
    for cls_id, (weight, cls_freq) in enumerate(zip(weights, freq), start=1):
        name = config.CLASS_NAMES.get(cls_id, f"cls{cls_id}")
        print(f"  [{cls_id}] {name:<22} freq={cls_freq:.2%}  weight={weight:.3f}")
    return weights


def compute_dynamic_weights(mask_dir:     str,
                             num_classes:  int,
                             ignore_index: int,
                             max_samples:  int   = 500,
                             smoothing:    float = 0.15) -> list:
    """
    Inverse-frequency weights: w_c = (1-smooth)/freq_c + smooth.
    Call before train() and replace config.CLASS_WEIGHTS with the result.

    Return a list of length num_classes, corresponding to classes 1..num_classes.
    """
    files  = sorted(glob.glob(os.path.join(mask_dir, "*.png")))[:max_samples]
    if not files:
        print("[Warning] No mask files found; using default weights")
        return config.CLASS_WEIGHTS

    counts = np.zeros(num_classes + 1, dtype=np.int64)
    for f in files:
        m = np.array(Image.open(f))
        for c in range(num_classes + 1):
            counts[c] += (m == c).sum()

    valid_counts = counts[1:num_classes + 1].astype(np.float64)
    total        = valid_counts.sum()
    if total == 0:
        return [1.0] * num_classes

    freq    = valid_counts / total
    weights = (1.0 - smoothing) / np.maximum(freq, 1e-8) + smoothing
    weights = (weights / weights.mean()).tolist()

    print(f"[Class weights] Computed from {len(files)} masks:")
    for i, (w, f) in enumerate(zip(weights, freq)):
        cls  = i + 1
        name = config.CLASS_NAMES.get(cls, f"cls{cls}")
        print(f"  [{cls}] {name:<22} frequency={f:.2%}  weight={w:.3f}")
    return weights


# =================================================================
# Combined loss
# =================================================================
class ImprovedLoss(nn.Module):
    """
    Focal(0.3) + Dice(0.3) + Lovász(0.4) with per-pixel LCI weighting.

    class_weights : (NUM_CLASSES+1,) class-level frequency balancing
    lci_weight    : per-pixel (H, W) tensor in [0,1]
                    = LCI_raw (geometric purity from prepare_data.py)
                    Pure pixels (LCI=1.0) contribute fully.
                    Mixed pixels (LCI=0.4) contribute proportionally.
                    Unlabeled pixels (IGNORE_INDEX) are excluded by mask.

    Combined effect of class_weights × LCI:
      - Rare classes (Dead trees) are upweighted by class_weights
      - Pure Dead tree pixels get full LCI weight
      - Dead/Living boundary pixels keep full LCI weight
        (boundary pixels ARE the critical training signal)
    """
    def __init__(self,
                 class_weights: list = None,
                 device:        str  = "cpu"):
        super().__init__()
        w = (torch.tensor(class_weights, dtype=torch.float32).to(device)
             if class_weights is not None else None)
        ign = config.IGNORE_INDEX

        self.class_weights = w
        self.focal  = FocalLoss(gamma=2.0, weight=w, ignore_index=ign)
        self.dice   = DiceLoss(ignore_index=ign)
        self.lovasz = LovaszLoss(ignore_index=ign)

    def forward(self,
                pred:         torch.Tensor,
                target:       torch.Tensor,
                lci_weight:   torch.Tensor = None,
                soft_targets: torch.Tensor = None,
                soft_valid_mask: torch.Tensor = None) -> torch.Tensor:
        """
        pred         : (B, C, H, W) logits
        target       : (B, H, W) int hard class labels  (for Dice + Lovász)
        lci_weight   : (B, H, W) float [0,1] per-pixel LCI weight
                       None → uniform weight
        soft_targets : (B, NUM_CLASSES+1, H, W) float soft label probabilities
                       from prepare_data_softlabel.py
                       None → fall back to hard label cross-entropy

        Loss design:
          When soft_targets available:
            soft_CE = -sum(soft_targets * log_softmax(pred))  per pixel
            This is KL divergence between soft_targets and predicted distribution.
            Pixel [0.65, 0.35, 0, ...] trains the model to output 0.65 for Dead,
            0.35 for Living — not a hard 0/1.

          When only hard labels:
            Standard LCI-weighted Focal loss.

          Dice + Lovász always use hard labels (they compute region statistics).
        """
        import torch.nn.functional as F

        # ── Dice and Lovász (always use hard labels) ────────────────────
        dice_loss   = self.dice(  pred, target)
        lovasz_loss = self.lovasz(pred, target)

        # ── Pixel-level loss (soft CE or focal) ─────────────────────────
        valid_mask = (target != config.IGNORE_INDEX)          # (B,H,W)
        if lci_weight is not None:
            w = lci_weight * valid_mask.float()
        else:
            w = valid_mask.float()

        def focal_pixel_map():
            log_probs = F.log_softmax(pred, dim=1)
            gather_target = target.clone()
            gather_target[~valid_mask] = 0
            log_pt = log_probs.gather(1, gather_target.unsqueeze(1)).squeeze(1)
            pt = log_pt.exp()
            if self.class_weights is not None:
                alpha = self.class_weights[gather_target]
            else:
                alpha = torch.ones_like(pt)
            return -alpha * (1 - pt) ** 2 * log_pt

        if soft_targets is not None:
            # Soft cross-entropy (KL divergence form)
            # soft_targets: (B, C, H, W) — class probabilities per pixel
            log_probs  = F.log_softmax(pred, dim=1)           # (B, C, H, W)
            if self.class_weights is not None:
                class_w = self.class_weights.view(1, -1, 1, 1)
                soft_ce = -(soft_targets * class_w * log_probs).sum(dim=1)
            else:
                soft_ce = -(soft_targets * log_probs).sum(dim=1)
            if soft_valid_mask is None:
                pixel_map = soft_ce
            else:
                soft_valid = soft_valid_mask.to(pred.device).reshape(-1, 1, 1).bool()
                if soft_valid.all():
                    pixel_map = soft_ce
                elif (~soft_valid).all():
                    pixel_map = focal_pixel_map()
                else:
                    pixel_map = torch.where(soft_valid, soft_ce, focal_pixel_map())
            pixel_loss = (pixel_map * w).sum() / (w.sum() + 1e-8)

        else:
            # Standard LCI-weighted Focal loss
            pixel_loss = (focal_pixel_map() * w).sum() / (w.sum() + 1e-8)

        # ── Combine ─────────────────────────────────────────────────────
        return (0.3 * pixel_loss +
                0.3 * dice_loss  +
                0.4 * lovasz_loss)
