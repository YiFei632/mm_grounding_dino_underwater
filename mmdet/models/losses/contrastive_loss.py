# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.registry import MODELS


@MODELS.register_module()
class ContrastiveLoss(nn.Module):
    """CLIP-style contrastive loss for multi-modal learning.

    This loss implements the InfoNCE objective used in CLIP, which aligns
    features from two modalities (e.g., sonar-image, sonar-text) by maximizing
    similarity between paired samples and minimizing similarity between
    unpaired samples within a batch.

    Args:
        temperature (float): Temperature parameter for scaling logits.
            Lower values make the distribution sharper. Defaults to 0.07.
        learnable_temperature (bool): Whether temperature is a learnable
            parameter. Defaults to False.
    """

    def __init__(self,
                 temperature: float = 0.07,
                 learnable_temperature: bool = False):
        super().__init__()
        if learnable_temperature:
            self.temperature = nn.Parameter(torch.ones([]) * temperature)
        else:
            self.register_buffer('temperature', torch.tensor(temperature))

    def forward(self,
                feat_a: torch.Tensor,
                feat_b: torch.Tensor,
                mask_a: torch.Tensor = None,
                mask_b: torch.Tensor = None) -> torch.Tensor:
        """Compute bidirectional contrastive loss between two modalities.

        Args:
            feat_a (Tensor): Features of modality A, shape (bs, num_tokens_a, dim)
            feat_b (Tensor): Features of modality B, shape (bs, num_tokens_b, dim)
            mask_a (Tensor, optional): Padding mask for feat_a, shape (bs, num_tokens_a).
                True for valid tokens, False for padding. Defaults to None.
            mask_b (Tensor, optional): Padding mask for feat_b, shape (bs, num_tokens_b).
                True for valid tokens, False for padding. Defaults to None.

        Returns:
            Tensor: Contrastive loss value (scalar).
        """
        # 1. Global pooling with mask consideration
        feat_a_global = self._masked_pool(feat_a, mask_a)  # (bs, dim)
        feat_b_global = self._masked_pool(feat_b, mask_b)  # (bs, dim)

        # 2. L2 normalize features
        feat_a_global = F.normalize(feat_a_global, dim=-1)
        feat_b_global = F.normalize(feat_b_global, dim=-1)

        # 3. Compute similarity matrix (cosine similarity after normalization)
        logits = feat_a_global @ feat_b_global.T / self.temperature  # (bs, bs)

        # 4. Labels: diagonal elements are positive pairs
        labels = torch.arange(logits.shape[0], device=logits.device)

        # 5. Bidirectional cross-entropy loss
        # A -> B: for each sample in A, find matching sample in B
        loss_a2b = F.cross_entropy(logits, labels)
        # B -> A: for each sample in B, find matching sample in A
        loss_b2a = F.cross_entropy(logits.T, labels)

        # Average bidirectional losses
        loss = (loss_a2b + loss_b2a) / 2
        return loss

    def _masked_pool(self, features: torch.Tensor, mask: torch.Tensor = None):
        """Masked mean pooling over the token dimension.

        Args:
            features (Tensor): Input features, shape (bs, num_tokens, dim)
            mask (Tensor, optional): Boolean mask, shape (bs, num_tokens).
                True for valid tokens, False for padding. Defaults to None.

        Returns:
            Tensor: Pooled features, shape (bs, dim)
        """
        if mask is None:
            # No mask provided, use simple mean pooling
            return features.mean(dim=1)

        # Convert boolean mask to float: True -> 1.0, False -> 0.0
        mask = mask.float()  # (bs, num_tokens)

        # Weighted sum: only sum over valid tokens
        masked_features = features * mask.unsqueeze(-1)  # (bs, num_tokens, dim)
        sum_features = masked_features.sum(dim=1)  # (bs, dim)

        # Normalize by number of valid tokens to get mean
        num_valid = mask.sum(dim=1, keepdim=True).clamp(min=1e-6)  # (bs, 1)
        pooled = sum_features / num_valid  # (bs, dim)

        return pooled
