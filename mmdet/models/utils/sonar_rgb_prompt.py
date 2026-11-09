"""Class-conditioned sonar evidence transported to an RGB feature grid."""
import math

import torch
from torch import nn
import torch.nn.functional as F


class SonarRGBPrompt(nn.Module):
    def __init__(self, embed_dims=256, temperature=0.07):
        super().__init__()
        if temperature <= 0:
            raise ValueError('temperature must be positive')
        self.temperature = temperature
        self.sonar_features = nn.Sequential(
            nn.Conv2d(embed_dims, embed_dims, 3, padding=1),
            nn.GroupNorm(8, embed_dims), nn.GELU(),
            nn.Conv2d(embed_dims, embed_dims, 1))
        self.rgb_match = nn.Linear(embed_dims, embed_dims)
        self.sonar_match = nn.Linear(embed_dims, embed_dims)
        self.sonar_class = nn.Linear(embed_dims, embed_dims)
        self.text_class = nn.Linear(embed_dims, embed_dims)
        self.null_logit = nn.Parameter(torch.zeros(()))
        self.class_bias = nn.Parameter(torch.tensor(-2.0))

    def forward(self, rgb, sonar_grid, classes, class_valid,
                rgb_mask=None, sonar_mask=None):
        sonar = self.sonar_features(sonar_grid).flatten(2).transpose(1, 2)
        # FP32 probabilities avoid underflow in log priors and auxiliary BCE.
        r = F.normalize(self.rgb_match(rgb).float(), dim=-1)
        s = F.normalize(self.sonar_match(sonar).float(), dim=-1)
        score = r @ s.transpose(1, 2) / self.temperature
        if sonar_mask is None:
            sonar_mask = torch.zeros(sonar.shape[:2], dtype=torch.bool,
                                     device=sonar.device)
        count = (~sonar_mask).sum(-1).clamp_min(1)
        score = score - count.float().log()[:, None, None]
        score = score.masked_fill(sonar_mask[:, None], float('-inf'))
        null = self.null_logit.float().expand(*score.shape[:2], 1)
        attention = torch.cat([score, null], -1).softmax(-1)
        association = attention[..., :-1]
        sc = F.normalize(self.sonar_class(sonar).float(), dim=-1)
        tc = F.normalize(self.text_class(classes).float(), dim=-1)
        logits = sc @ tc.transpose(1, 2) / self.temperature + self.class_bias
        evidence = logits.sigmoid() * class_valid[:, None].float()
        evidence = evidence.masked_fill(sonar_mask[..., None], 0)
        prompt = association @ evidence
        if rgb_mask is not None:
            prompt = prompt.masked_fill(rgb_mask[..., None], 0)
        return prompt, logits, sonar, attention[..., -1]


def prompt_token_bias(prompt, class_token_map, width):
    """Broadcast class evidence to its tokens, with bounded overlap handling."""
    membership = (class_token_map > 0).to(prompt.dtype)
    bias = prompt @ membership
    bias = bias / membership.sum(1).clamp_min(1)[:, None]
    if bias.shape[-1] > width:
        raise ValueError('Text length exceeds classification width')
    return F.pad(bias, (0, width - bias.shape[-1]))


def box_class_targets(instances, image_size, grid_size, classes, device):
    """Box/cell overlap masks; coordinates are in the specified image canvas."""
    h, w = grid_size
    ih, iw = image_size
    targets = torch.zeros(h, w, classes, device=device)
    for box, label in zip(instances.bboxes, instances.labels):
        c = int(label)
        if not 0 <= c < classes:
            raise ValueError('GT label is outside the prompt class list')
        x1, y1, x2, y2 = box.detach().tolist()
        if x2 <= x1 or y2 <= y1:
            continue
        xa, xb = max(0, math.floor(x1 / iw * w)), min(w, math.ceil(x2 / iw * w))
        ya, yb = max(0, math.floor(y1 / ih * h)), min(h, math.ceil(y2 / ih * h))
        targets[ya:yb, xa:xb, c] = 1
    return targets.flatten(0, 1)


def spatial_bce_dice(prediction, target, valid, logits=False):
    """Ignore padding and missing classes in both BCE and Dice."""
    prediction, target = prediction.float(), target.float()
    valid = valid.expand_as(target).float()
    if logits:
        bce = F.binary_cross_entropy_with_logits(prediction, target, reduction='none')
        prob = prediction.sigmoid()
    else:
        prob = prediction.clamp(1e-6, 1 - 1e-6)
        bce = F.binary_cross_entropy(prob, target, reduction='none')
    bce = (bce * valid).sum() / valid.sum().clamp_min(1)
    intersection = (prob * target * valid).sum(1)
    dice = 1 - (2 * intersection + 1) / (((prob + target) * valid).sum(1) + 1)
    active = valid.sum(1) > 0
    return bce + (dice * active).sum() / active.sum().clamp_min(1)
