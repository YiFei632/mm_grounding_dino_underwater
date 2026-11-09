import torch
from torch import nn
import torch.nn.functional as F
from mmdet.registry import MODELS

@MODELS.register_module()
class MDSFF(nn.Module):
    """Multi-modal deformable self-aligned fine-grained fusion.

    The offset field samples the reference modality around each sensed pixel,
    then a gated differential fusion keeps complementary information.
    """
    def __init__(self, channels=256, groups=8, kernel_size=3):
        super().__init__()
        self.offset = nn.Conv2d(channels * 2, 2 * kernel_size * kernel_size, 3, padding=1)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.gate = nn.Sequential(nn.Conv2d(channels * 2, channels, 1), nn.Sigmoid())
        self.out = nn.Conv2d(channels * 2, channels, 1)
        self.kernel_size = kernel_size

    def forward(self, sensed, reference):
        if sensed.shape[-2:] != reference.shape[-2:]:
            reference = F.interpolate(reference, sensed.shape[-2:], mode='bilinear', align_corners=False)
        b, c, h, w = sensed.shape
        # Offset-guided local sampling (grid_sample is differentiable and CUDA friendly).
        offsets = self.offset(torch.cat([sensed, reference], 1)).mean(1, keepdim=True)
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=sensed.device),
                                torch.linspace(-1, 1, w, device=sensed.device), indexing='ij')
        grid = torch.stack([xx, yy], -1).expand(b, -1, -1, -1).clone()
        grid[..., 0] += offsets[:, 0].tanh() / max(w, 1)
        grid[..., 1] += offsets[:, 0].tanh() / max(h, 1)
        aligned = F.grid_sample(reference, grid, mode='bilinear', padding_mode='border', align_corners=True)
        q, k, v = self.q(sensed), self.k(aligned), self.v(aligned)
        attn = torch.sigmoid((q * k) / (c ** 0.5))
        diff = torch.abs(sensed - aligned)
        gate = self.gate(torch.cat([diff, sensed + aligned], 1))
        return self.out(torch.cat([sensed + gate * attn * v, aligned + (1-gate) * diff], 1))

@MODELS.register_module()
class MIFM(nn.Module):
    """Lightweight instance/spatial matching fusion used after RPN features."""
    def __init__(self, channels=256):
        super().__init__()
        self.proj = nn.Conv2d(channels * 2, channels, 1)
        self.temperature = channels ** -0.5

    def forward(self, sensed, reference):
        if sensed.shape[-2:] != reference.shape[-2:]:
            reference = F.interpolate(reference, sensed.shape[-2:], mode='bilinear', align_corners=False)
        b, c, h, w = sensed.shape
        qs = F.adaptive_avg_pool2d(sensed, 1).flatten(1)
        kr = F.adaptive_avg_pool2d(reference, 1).flatten(1)
        score = torch.sigmoid((qs * kr).sum(1, keepdim=True) * self.temperature).view(b, 1, 1, 1)
        return self.proj(torch.cat([sensed, reference * score], 1))
