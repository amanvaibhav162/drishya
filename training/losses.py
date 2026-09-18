"""
DRISHYA SOTA Training: Loss Functions
Includes:
  - CoralLoss: Consistent Rank Logits (CORAL) for Ordinal Classification
  - SmoothL1 on continuous expected grade
  - Multi-Channel Dice Loss
  - Focal-Tversky Loss with beta=0.7 (punishing False Negatives on tiny microaneurysms)
  - Unified Multi-Task Loss with partial mask supervision support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def grade_to_coral_target(grades: torch.Tensor, num_classes: int = 5) -> torch.Tensor:
    """
    Converts integer grades [B] into binary ordinal targets [B, num_classes - 1].
    Grade 0 -> [0, 0, 0, 0]
    Grade 1 -> [1, 0, 0, 0]
    Grade 2 -> [1, 1, 0, 0]
    Grade 3 -> [1, 1, 1, 0]
    Grade 4 -> [1, 1, 1, 1]
    """
    num_thresholds = num_classes - 1
    # grades shape [B, 1]
    g = grades.unsqueeze(1)
    thresholds = torch.arange(num_thresholds, device=grades.device).unsqueeze(0)
    # 1 if grade > threshold else 0
    return (g > thresholds).float()


def coral_logits_to_expected_grade(logits: torch.Tensor) -> torch.Tensor:
    """
    Converts CORAL binary logits [B, 4] to expected continuous grade E[y] in [0, 4].
    E[y] = sum_{k=0}^{K-2} sigmoid(logits_k)
    """
    probs = torch.sigmoid(logits)
    return probs.sum(dim=-1)


def coral_logits_to_class_probs(logits: torch.Tensor) -> torch.Tensor:
    """
    Converts CORAL binary logits [B, 4] into categorical probabilities [B, 5].
    P(y = 0) = 1 - P(y > 0)
    P(y = k) = P(y > k-1) - P(y > k)
    P(y = 4) = P(y > 3)
    """
    probs = torch.sigmoid(logits)  # [B, 4]
    b = logits.shape[0]
    device = logits.device

    p_greater = probs  # P(y > 0), P(y > 1), P(y > 2), P(y > 3)
    p0 = 1.0 - p_greater[:, 0:1]
    p1 = torch.clamp(p_greater[:, 0:1] - p_greater[:, 1:2], min=0.0)
    p2 = torch.clamp(p_greater[:, 1:2] - p_greater[:, 2:3], min=0.0)
    p3 = torch.clamp(p_greater[:, 2:3] - p_greater[:, 3:4], min=0.0)
    p4 = p_greater[:, 3:4]

    class_probs = torch.cat([p0, p1, p2, p3, p4], dim=-1)
    # Normalize to ensure sum to 1
    class_probs = class_probs / (class_probs.sum(dim=-1, keepdim=True) + 1e-7)
    return class_probs


class CoralLoss(nn.Module):
    """
    Consistent Rank Logits (CORAL) Loss for Ordinal Regression.
    """
    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: [B, num_classes - 1]
        targets: [B] integer class labels (0 to num_classes - 1)
        """
        binary_targets = grade_to_coral_target(targets, self.num_classes)
        return F.binary_cross_entropy_with_logits(logits, binary_targets)


class MultiChannelDiceLoss(nn.Module):
    """
    Soft Dice loss across 4 lesion channels (MA, EX, HE, SE).
    """
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        logits: [B, C, H, W]
        targets: [B, C, H, W]
        mask_weight: [B] binary tensor indicating whether ground truth mask exists
        """
        probs = torch.sigmoid(logits)
        batch_size, num_channels = probs.shape[:2]

        probs = probs.view(batch_size, num_channels, -1)
        targets = targets.view(batch_size, num_channels, -1)

        intersection = (probs * targets).sum(dim=-1)
        cardinality = probs.sum(dim=-1) + targets.sum(dim=-1)

        dice_per_channel = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        channel_loss = 1.0 - dice_per_channel  # [B, C]

        if mask_weight is not None:
            # Only average over samples with valid ground truth masks
            valid_mask = mask_weight.unsqueeze(1)  # [B, 1]
            valid_count = valid_mask.sum()
            if valid_count > 0:
                loss = (channel_loss * valid_mask).sum() / (valid_count * num_channels)
            else:
                loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        else:
            loss = channel_loss.mean()

        return loss


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss.
    Beta = 0.7 places higher weight on False Negatives (penalizes missed microaneurysms).
    Alpha = 0.3.
    Gamma = 1.33 focuses on hard, ambiguous lesion boundaries.
    """
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, gamma: float = 1.33, smooth: float = 1e-5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        batch_size, num_channels = probs.shape[:2]

        probs = probs.view(batch_size, num_channels, -1)
        targets = targets.view(batch_size, num_channels, -1)

        true_pos = (probs * targets).sum(dim=-1)
        false_pos = (probs * (1.0 - targets)).sum(dim=-1)
        false_neg = ((1.0 - probs) * targets).sum(dim=-1)

        tversky = (true_pos + self.smooth) / (true_pos + self.alpha * false_pos + self.beta * false_neg + self.smooth)
        loss = torch.pow(1.0 - tversky, self.gamma)  # [B, C]

        if mask_weight is not None:
            valid_mask = mask_weight.unsqueeze(1)
            valid_count = valid_mask.sum()
            if valid_count > 0:
                return (loss * valid_mask).sum() / (valid_count * num_channels)
            else:
                return torch.tensor(0.0, device=logits.device, requires_grad=True)

        return loss.mean()


class DRISHYAMultiTaskLoss(nn.Module):
    """
    Comprehensive multi-task loss combining:
      - CORAL ordinal classification loss
      - Smooth L1 expected grade loss
      - Multi-channel Dice loss
      - Focal Tversky lesion loss
    Gracefully handles mixed datasets (with and without pixel lesion masks).
    """
    def __init__(
        self,
        num_classes: int = 5,
        lambda_ordinal: float = 1.0,
        lambda_smooth_l1: float = 0.25,
        lambda_dice: float = 0.50,
        lambda_tversky: float = 0.50
    ):
        super().__init__()
        self.num_classes = num_classes
        self.coral_loss = CoralLoss(num_classes=num_classes)
        self.smooth_l1 = nn.SmoothL1Loss()
        self.dice_loss = MultiChannelDiceLoss()
        self.tversky_loss = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=1.33)

        self.lambda_ordinal = lambda_ordinal
        self.lambda_smooth_l1 = lambda_smooth_l1
        self.lambda_dice = lambda_dice
        self.lambda_tversky = lambda_tversky

    def forward(
        self,
        pred_logits: torch.Tensor,
        pred_masks: torch.Tensor,
        target_grades: torch.Tensor,
        target_masks: Optional[torch.Tensor] = None,
        has_mask: Optional[torch.Tensor] = None,
        soft_targets: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, dict]:
        """
        pred_logits: [B, 4] for CORAL (or [B, 5] if standard cross entropy)
        pred_masks:  [B, 4, H, W]
        target_grades: [B] integer 0..4
        target_masks:  [B, 4, H, W] (optional)
        has_mask:      [B] 1.0 if sample has ground truth lesion mask, 0.0 otherwise
        soft_targets:  [B, 5] (optional soft probability targets from teacher ensemble)
        """
        # 1. Ordinal loss
        loss_ord = self.coral_loss(pred_logits, target_grades)

        # 2. Smooth L1 on continuous expected grade
        expected_grade = coral_logits_to_expected_grade(pred_logits)
        loss_reg = self.smooth_l1(expected_grade, target_grades.float())

        total_loss = self.lambda_ordinal * loss_ord + self.lambda_smooth_l1 * loss_reg

        loss_dice = torch.tensor(0.0, device=pred_logits.device)
        loss_tv = torch.tensor(0.0, device=pred_logits.device)

        # 3. Lesion segmentation losses (only if target masks available)
        if target_masks is not None and (has_mask is None or has_mask.sum() > 0):
            loss_dice = self.dice_loss(pred_masks, target_masks, mask_weight=has_mask)
            loss_tv = self.tversky_loss(pred_masks, target_masks, mask_weight=has_mask)
            total_loss = total_loss + self.lambda_dice * loss_dice + self.lambda_tversky * loss_tv

        breakdown = {
            "loss_total": total_loss.item(),
            "loss_ordinal": loss_ord.item(),
            "loss_smooth_l1": loss_reg.item(),
            "loss_dice": loss_dice.item() if isinstance(loss_dice, torch.Tensor) else loss_dice,
            "loss_tversky": loss_tv.item() if isinstance(loss_tv, torch.Tensor) else loss_tv,
        }

        return total_loss, breakdown


class DistillationLoss(nn.Module):
    """
    Combined loss for knowledge distillation:
      L_total = (1 - alpha) * L_hard_mtl + alpha * L_soft_kl
    """
    def __init__(self, mtl_criterion: DRISHYAMultiTaskLoss, alpha: float = 0.5, temperature: float = 2.0):
        super().__init__()
        self.mtl_criterion = mtl_criterion
        self.alpha = alpha
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(
        self,
        pred_logits: torch.Tensor,
        pred_masks: torch.Tensor,
        target_grades: torch.Tensor,
        target_masks: Optional[torch.Tensor] = None,
        has_mask: Optional[torch.Tensor] = None,
        soft_targets: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, dict]:
        hard_loss, breakdown = self.mtl_criterion(
            pred_logits, pred_masks, target_grades, target_masks, has_mask
        )

        if soft_targets is not None:
            student_probs = coral_logits_to_class_probs(pred_logits / self.temperature)
            log_student_probs = torch.log(student_probs + 1e-7)
            soft_loss = self.kl_div(log_student_probs, soft_targets) * (self.temperature ** 2)
            total_loss = (1.0 - self.alpha) * hard_loss + self.alpha * soft_loss
            breakdown["loss_soft_kd"] = soft_loss.item()
            breakdown["loss_total"] = total_loss.item()
            return total_loss, breakdown

        return hard_loss, breakdown

