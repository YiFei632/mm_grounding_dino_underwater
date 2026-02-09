import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import List
from mmdet.registry import MODELS

class GaussianBlurConv(object):
    def FastFilter(self, img, sigma):
        if sigma > 300: sigma = 300
        kernel_size = round(sigma * 3 * 2 + 1) | 1
        if kernel_size < 3: return img
        if kernel_size < 10:
            return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        else:
            if img.shape[1] < 2 or img.shape[0] < 2: return img
            sub_img = cv2.pyrDown(img)
            sub_img = self.FastFilter(sub_img, sigma / 2.0)
            res_img = cv2.resize(sub_img, (img.shape[1], img.shape[0]))
            return res_img

    def __call__(self, x, sigma):
        return self.FastFilter(x, sigma)

@MODELS.register_module()
class MSREnhancement(nn.Module):
    def __init__(self, 
                 sigma: List[float] = [30, 150, 300], 
                 restore_factor: float = 2.0, 
                 color_gain: float = 10.0, 
                 gain: float = 270.0, 
                 offset: float = 128.0):
        super().__init__()
        self.sigma = sigma
        self.restore_factor = restore_factor
        self.color_gain = color_gain
        self.gain = gain
        self.offset = offset
        self.gaussian_conv = GaussianBlurConv()
        # 冻结参数，因为这主要是传统算法
        for p in self.parameters():
            p.requires_grad = False

    def _gaussian_blur_optimized(self, x_np, sigma):
        # x_np: [H, W, 3]
        return self.gaussian_conv(x_np, sigma)

    def _ssr(self, img, sigma):
        # 这里为了简化计算，我们在 CPU numpy 上做 blur，然后转回 tensor 计算 log
        # 注意：这会打断梯度流，但原论文中 Enhancer 似乎不参与反向传播更新（或者作为预处理）
        # 如果需要梯度，必须用 Kornia 等库重写 GaussianBlur
        dev = img.device
        img_np = img.detach().cpu().numpy() # [H, W, 3]
        blur_np = self._gaussian_blur_optimized(img_np, sigma)
        blur_tensor = torch.from_numpy(blur_np).to(dev)
        return torch.log10(img + 1e-6) - torch.log10(blur_tensor + 1e-6)

    def _msr(self, img):
        retinex = torch.zeros_like(img)
        for sig in self.sigma:
            retinex += self._ssr(img, sig)
        return retinex / len(self.sigma)

    def forward_single(self, img_tensor):
        # img_tensor: [3, H, W], normalized 0-1
        img = img_tensor.permute(1, 2, 0) * 255.0 # [H, W, 3]
        img_float = img + 1.0
        
        retinex = self._msr(img_float)
        
        img_sum = torch.sum(img_float, dim=2, keepdim=True)
        color_restoration = torch.log10((img_float * self.restore_factor / (img_sum + 1e-6)) + 1.0)
        img_merge = retinex * color_restoration * self.color_gain
        enhanced = img_merge * self.gain + self.offset
        
        enhanced = torch.clamp(enhanced, 0, 255) / 255.0
        return enhanced.permute(2, 0, 1) # [3, H, W]

    def forward(self, x):
        # x: [B, 3, H, W]
        out = []
        for i in range(x.size(0)):
            out.append(self.forward_single(x[i]))
        return torch.stack(out)