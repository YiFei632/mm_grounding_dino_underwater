# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmdet.registry import MODELS
from .utils import weight_reduce_loss


@MODELS.register_module()
class RoIContrastiveFocalLoss(nn.Module):
    """RoI-level Contrastive Focal Loss for region-text alignment.

    This loss implements focal loss on similarity scores between RoI features
    and text features, designed for pretraining visual-language encoders.

    Args:
        alpha (float): Weighting factor in range (0,1) to balance
            positive vs negative examples. Default: 0.25.
        gamma (float): Exponent of the modulating factor (1 - p_t)^gamma
            to balance easy vs hard examples. Default: 2.0.
        temperature (float): Temperature parameter for scaling similarity.
            Default: 0.07.
        loss_weight (float): Weight of the loss. Default: 1.0.
        reduction (str): Options are "none", "mean" and "sum".
            Default: "mean".
    """

    def __init__(self,
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 temperature: float = 0.07,
                 loss_weight: float = 1.0,
                 reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.temperature = temperature
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self,
                similarity: Tensor,
                target: Tensor,
                weight: Tensor = None,
                avg_factor: int = None) -> Tensor:
        """Forward function.

        Args:
            similarity (Tensor): Similarity scores between RoI features and
                text features, shape (N, K), where N is number of RoIs and
                K is number of classes.
            target (Tensor): Ground truth labels, shape (N, K), binary values
                where 1 indicates positive match and 0 indicates negative.
            weight (Tensor, optional): Sample-wise loss weight.
            avg_factor (int, optional): Average factor for loss normalization.

        Returns:
            Tensor: Calculated loss.
        """
        # Scale similarity by temperature
        similarity = similarity / self.temperature

        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(similarity)

        # Ensure target is same type as probs
        target = target.type_as(probs)

        # Calculate focal weight: (1 - p_t)^gamma
        # For positive samples: p_t = probs, focal_weight = (1 - probs)^gamma
        # For negative samples: p_t = 1 - probs, focal_weight = probs^gamma
        pt = (1 - probs) * target + probs * (1 - target)
        focal_weight = pt.pow(self.gamma)

        # Calculate alpha weight
        # For positive samples: alpha_t = alpha
        # For negative samples: alpha_t = 1 - alpha
        alpha_weight = self.alpha * target + (1 - self.alpha) * (1 - target)

        # Combine focal weight and alpha weight
        weight_factor = alpha_weight * focal_weight

        # Calculate binary cross entropy loss
        bce_loss = F.binary_cross_entropy_with_logits(
            similarity, target, reduction='none')

        # Apply focal weight
        loss = weight_factor * bce_loss

        # Apply sample-wise weight if provided
        if weight is not None:
            if weight.shape != loss.shape:
                if weight.size(0) == loss.size(0):
                    weight = weight.view(-1, 1)
                else:
                    assert weight.numel() == loss.numel()
                    weight = weight.view(loss.size(0), -1)
            assert weight.ndim == loss.ndim

        # Reduce loss
        loss = weight_reduce_loss(loss, weight, self.reduction, avg_factor)

        return loss * self.loss_weight
