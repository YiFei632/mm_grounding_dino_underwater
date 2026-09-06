"""
SCANet组件 - SCAM (Sonar-Camera Attention Module)
跨模态注意力融合模块
"""

import torch
import torch.nn as nn


class SCAM(nn.Module):
    """
    Sonar-Camera Attention Module
    用于RGB和Sonar特征的跨模态融合
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Cross-attention components
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        简化版本: 对特征进行自注意力处理
        完整版本需要分离RGB和Sonar特征进行交叉注意力

        Args:
            x: (B, N, C) 输入特征
        Returns:
            x: (B, N, C) 融合后的特征
        """
        B, N, C = x.shape

        # 简化版: 使用自注意力
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
