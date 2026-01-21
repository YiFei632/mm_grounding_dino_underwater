import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.registry import MODELS
from mmdet.models.dense_heads import RepPointsHead

# =================================================================
# 1. ASPP Module (空洞空间金字塔池化)
# 对应论文 Section 3.4.1 和 Eq.(5)
# =================================================================
class ASPPModule(nn.Module):
    """
    ASPP (Atrous Spatial Pyramid Pooling) as described in PE-Transformer.
    Reference: Eq (5) in the paper.
    """
    def __init__(self, in_channels, out_channels, dilations=(1, 6, 12, 18)):
        super().__init__()
        self.aspp_convs = nn.ModuleList()
        
        # 1. 不同膨胀率的空洞卷积分支
        for dilation in dilations:
            self.aspp_convs.append(
                ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    padding=dilation,
                    dilation=dilation,
                    norm_cfg=dict(type='BN'), # GroupNorm 通常比 BN 在 Head 中更稳
                    act_cfg=dict(type='ReLU')
                )
            )
            
        # 2. 全局平均池化分支 (Global Average Pooling)
        # 论文 Eq.(5) 提到了聚合不同尺度的上下文
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            ConvModule(
                in_channels,
                out_channels,
                1,
                stride=1,
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='ReLU')
            )
        )
        
        # 3. 融合层
        self.bottleneck = ConvModule(
            out_channels * (len(dilations) + 1), # +1 for global pool
            out_channels,
            1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU')
        )

    def forward(self, x):
        outs = []
        # 处理空洞卷积分支
        for aspp_conv in self.aspp_convs:
            outs.append(aspp_conv(x))
            
        # 处理全局池化分支 (并上采样回原尺寸)
        global_feat = self.global_avg_pool(x)
        global_feat = torch.nn.functional.interpolate(
            global_feat, size=x.shape[2:], mode='bilinear', align_corners=True)
        outs.append(global_feat)
        
        # 拼接并融合
        out = torch.cat(outs, dim=1)
        out = self.bottleneck(out)
        return out

# =================================================================
# 2. PE-Transformer Head (Decoder)
# 对应论文 Figure 4 (Shared-Head + ASAA)
# =================================================================
@MODELS.register_module()
class PETransformerHead(RepPointsHead):
    """
    PE-Transformer Decoder Head.
    Inherits from RepPointsHead to implement 'Adaptive Point Representation'.
    Adds ASPP module before feature prediction.
    """
    def __init__(self, 
                 in_channels=256, 
                 feat_channels=256, 
                 point_feat_channels=256, 
                 num_classes=80,
                 **kwargs):
        
        # 初始化父类 RepPointsHead
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            feat_channels=feat_channels,
            point_feat_channels=point_feat_channels,
            **kwargs
        )
        
        # 【关键修改】 插入 ASPP 模块
        # 论文引用: "ASPP component generates these feature maps"
        self.aspp = ASPPModule(in_channels, feat_channels)

    def forward_single(self, x):
        """
        Forward feature of a single scale level.
        x: Feature map from Neck (e.g., P3)
        """
        # 1. 先通过 ASPP 增强特征上下文
        # 这是 PE-Transformer 与普通 RepPoints 的主要区别
        dcn_base_offset = None
        dcn_cls_offset = None
        
        # 将输入特征 x 传入 ASPP
        x_aspp = self.aspp(x)
        
        # 2. 后续流程复用 RepPoints 的逻辑 (Shared Convs -> Cls/Reg)
        # 论文 [cite: 261] 提到 "two conversion functions":
        #   - 1st: generate adaptive point sets (refine_pts_init)
        #   - 2nd: fine-tune detection frame (refine_pts_refine)
        # 这完全对应 RepPoints 的两次回归过程。
        
        return super().forward_single(x_aspp)