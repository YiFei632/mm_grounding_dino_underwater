import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.registry import MODELS


@MODELS.register_module()
class SonarFeatureSelector(nn.Module):
    """Lightweight text-conditioned spatial selector for sonar features."""
    def __init__(self, in_channels=2048, hidden_channels=128, embed_dims=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(8, hidden_channels), nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(8, hidden_channels), nn.GELU())
        self.feature = nn.Conv2d(hidden_channels, embed_dims, 1)
        self.score = nn.Sequential(nn.Linear(embed_dims * 2, embed_dims),
                                   nn.GELU(), nn.Linear(embed_dims, 1))

    def forward(self, sonar, text, text_mask=None):
        x = self.cnn(sonar)
        feat = self.feature(x)
        b, d, h, w = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)
        if text_mask is None:
            text_global = text.mean(1, keepdim=True)
        else:
            valid = text_mask.to(text.dtype).unsqueeze(-1)
            text_global = (text * valid).sum(1, keepdim=True) / valid.sum(
                1, keepdim=True).clamp_min(1)
        text_global = text_global.expand(-1, h * w, -1)
        logits = self.score(torch.cat([tokens, text_global], -1)).squeeze(-1)
        weights = logits.sigmoid()
        return tokens * weights.unsqueeze(-1), logits, weights


def sonar_box_mask(boxes, image_size, feature_size):
    """Rasterize sonar GT boxes into a feature-map foreground mask."""
    h, w = feature_size
    ih, iw = image_size
    mask = boxes.new_zeros((h, w))
    for box in boxes:
        x1, y1, x2, y2 = box
        x1, x2 = int(x1 / iw * w), int(x2 / iw * w)
        y1, y2 = int(y1 / ih * h), int(y2 / ih * h)
        mask[max(0, y1):min(h, y2 + 1), max(0, x1):min(w, x2 + 1)] = 1
    return mask
