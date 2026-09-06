"""
SCANet Vision Transformer Backbone for MMDetection
支持4通道输入 (RGB + Sonar)，用于水下目标检测
"""

import torch
import torch.nn as nn
import math
from mmengine.model import BaseModule
from mmdet.registry import MODELS

from .scanet_components import PatchEmbed, Block, SCAM


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """初始化函数"""
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


@MODELS.register_module()
class SCANetViT(BaseModule):
    """
    SCANet Vision Transformer for underwater RGB-Sonar detection

    简化版本: 接收4通道输入 (RGB + Sonar)，输出多尺度特征供FPN使用

    Args:
        img_size (int): 输入图像尺寸. Default: 224
        patch_size (int): Patch大小. Default: 16
        in_chans (int): 输入通道数 (4 for RGB+Sonar). Default: 4
        embed_dim (int): Patch嵌入维度. Default: 768
        depth (int): Transformer层数. Default: 12
        num_heads (int): 注意力头数. Default: 12
        mlp_ratio (float): MLP隐藏层倍数. Default: 4.0
        qkv_bias (bool): 是否使用qkv bias. Default: True
        drop_rate (float): Dropout rate. Default: 0.0
        attn_drop_rate (float): Attention dropout rate. Default: 0.0
        drop_path_rate (float): Stochastic depth rate. Default: 0.0
        rgbs_loc (list): SCAM融合层位置. Default: [3, 6, 9]
        out_indices (tuple): 输出特征层索引. Default: (3, 5, 7, 11)
        pretrained (str): 预训练权重路径. Default: None
        norm_eval (bool): 是否在eval模式freeze norm层. Default: False
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=4,  # RGB(3) + Sonar(1)
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        rgbs_loc=[3, 6, 9],  # SCAM融合层位置
        out_indices=(3, 5, 7, 11),  # 输出4个尺度的特征
        pretrained=None,
        norm_eval=False,
        init_cfg=None
    ):
        super().__init__(init_cfg=init_cfg)

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.out_indices = out_indices
        self.norm_eval = norm_eval

        # Patch Embedding: 4通道输入
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i]
            )
            for i in range(depth)
        ])

        # SCAM融合模块（在指定层插入）
        self.rgbs_loc = rgbs_loc
        self.scam_layers = nn.ModuleDict()
        for loc in rgbs_loc:
            if loc < depth:
                self.scam_layers[f'scam_{loc}'] = SCAM(
                    dim=embed_dim,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    attn_drop=attn_drop_rate,
                    proj_drop=drop_rate
                )

        # Normalization layers for output features
        self.norms = nn.ModuleDict()
        for idx in out_indices:
            if idx < depth:
                self.norms[f'norm_{idx}'] = nn.LayerNorm(embed_dim)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """权重初始化"""
        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_vit_weights)

    def _init_vit_weights(self, m):
        """ViT标准初始化"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def init_weights(self):
        """MMDetection的初始化接口"""
        # 先执行标准初始化
        self._init_weights()

        # 如果有预训练权重配置，手动加载
        if self.init_cfg is not None and self.init_cfg.get('type') == 'Pretrained':
            checkpoint = self.init_cfg.get('checkpoint')
            if checkpoint:
                self._load_pretrained_checkpoint(checkpoint)

    def _load_pretrained_checkpoint(self, checkpoint_path):
        """手动加载预训练权重"""
        import os
        from mmengine.runner import CheckpointLoader

        print(f"🔄 加载预训练权重: {checkpoint_path}")

        if not os.path.exists(checkpoint_path):
            print(f"⚠️  警告: checkpoint不存在: {checkpoint_path}")
            return

        # 加载checkpoint
        checkpoint = CheckpointLoader.load_checkpoint(checkpoint_path, map_location='cpu')

        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        # 处理pos_embed大小不匹配（移除class token对应的位置）
        if 'pos_embed' in state_dict:
            pos_embed_checkpoint = state_dict['pos_embed']
            pos_embed_model = self.pos_embed

            if pos_embed_checkpoint.shape != pos_embed_model.shape:
                print(f"   调整pos_embed: {pos_embed_checkpoint.shape} -> {pos_embed_model.shape}")

                # 如果checkpoint有197个位置（带class token），模型有196个（不带）
                if pos_embed_checkpoint.shape[1] == 197 and pos_embed_model.shape[1] == 196:
                    # 移除第一个位置（class token的位置）
                    state_dict['pos_embed'] = pos_embed_checkpoint[:, 1:, :]
                    print(f"   移除class token位置，新shape: {state_dict['pos_embed'].shape}")
                elif pos_embed_checkpoint.shape[1] == 196 and pos_embed_model.shape[1] == 197:
                    # 如果checkpoint没有class token但模型有，添加一个零位置
                    cls_pos = torch.zeros(1, 1, pos_embed_checkpoint.shape[2])
                    state_dict['pos_embed'] = torch.cat([cls_pos, pos_embed_checkpoint], dim=1)
                    print(f"   添加class token位置，新shape: {state_dict['pos_embed'].shape}")

        # 移除不需要的keys（如cls_token，因为检测任务不需要）
        keys_to_remove = []
        for key in state_dict.keys():
            if 'cls_token' in key or 'head' in key:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            state_dict.pop(key)

        if keys_to_remove:
            print(f"   移除不需要的keys: {keys_to_remove}")

        # 加载权重（允许部分匹配）
        msg = self.load_state_dict(state_dict, strict=False)

        print(f"✅ 权重加载完成")
        print(f"   匹配的keys: {len(state_dict) - len(msg.missing_keys)}")

        if msg.missing_keys:
            print(f"   缺失的keys: {len(msg.missing_keys)}")
            if len(msg.missing_keys) <= 10:
                for key in msg.missing_keys:
                    print(f"     - {key}")
            else:
                for key in msg.missing_keys[:5]:
                    print(f"     - {key}")
                print(f"     ... 还有 {len(msg.missing_keys) - 5} 个")

        if msg.unexpected_keys:
            print(f"   多余的keys: {len(msg.unexpected_keys)}")
            if len(msg.unexpected_keys) <= 10:
                for key in msg.unexpected_keys:
                    print(f"     - {key}")
            else:
                for key in msg.unexpected_keys[:5]:
                    print(f"     - {key}")
                print(f"     ... 还有 {len(msg.unexpected_keys) - 5} 个")

    def forward(self, x):
        """
        Args:
            x (Tensor): 输入图像，shape (B, 4, H, W)

        Returns:
            tuple: 多尺度特征 tuple
                   每个特征 shape (B, embed_dim, H', W')
        """
        B = x.shape[0]

        # Patch embedding: (B, 4, H, W) -> (B, N, embed_dim)
        x = self.patch_embed(x)

        # 添加position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # 通过Transformer blocks并收集输出
        outs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)

            # SCAM融合（如果在融合层）
            if i in self.rgbs_loc and f'scam_{i}' in self.scam_layers:
                x = self.scam_layers[f'scam_{i}'](x)

            # 收集输出特征
            if i in self.out_indices:
                # Normalize
                x_norm = self.norms[f'norm_{i}'](x)

                # Reshape: (B, N, C) -> (B, C, H, W)
                # N = (img_size / patch_size) ^ 2
                H = W = int(x_norm.shape[1] ** 0.5)
                x_out = x_norm.transpose(1, 2).reshape(B, self.embed_dim, H, W)
                outs.append(x_out)

        return tuple(outs)

    def train(self, mode=True):
        """重写train方法以支持norm_eval"""
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.LayerNorm):
                    m.eval()
        return self
