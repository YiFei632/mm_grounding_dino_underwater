import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmdet.registry import MODELS
from mmcv.cnn import ConvModule

# 1. Feature Aggregation Module (FAM)
class FAM(BaseModule):
    def __init__(self, low_channels, high_channels, out_channels, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        # 使用 ConvModule 自动处理 Conv+BN+ReLU
        self.low_conv = nn.Sequential(
            ConvModule(low_channels, out_channels, 3, padding=1, norm_cfg=dict(type='BN')),
            ConvModule(out_channels, out_channels, 1, norm_cfg=dict(type='BN'))
        )
        
        self.high_conv = nn.Sequential(
            ConvModule(high_channels, out_channels, 3, padding=1, norm_cfg=dict(type='BN')),
            ConvModule(out_channels, out_channels, 1, norm_cfg=dict(type='BN'))
        )
        
        self.out_conv = ConvModule(out_channels, out_channels, 3, padding=1, norm_cfg=dict(type='BN'))

    def forward(self, f_low, f_high):
        x_low = self.low_conv(f_low)
        
        x_high = self.high_conv(f_high)
        # 上采样对齐
        x_high = F.interpolate(x_high, size=f_low.shape[-2:], mode='bilinear', align_corners=True)
        
        return self.out_conv(x_low + x_high)

# 2. Path Attention Module (PAM)
class PAM(BaseModule):
    def __init__(self, in_channels, reduction=16, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.conv_reduce = ConvModule(in_channels, in_channels // reduction, 1, act_cfg=dict(type='ReLU'), norm_cfg=None)
        self.conv_expand = ConvModule(in_channels // reduction, in_channels, 1, act_cfg=dict(type='Sigmoid'), norm_cfg=None)

    def forward(self, x):
        identity = x
        # 模拟论文中的局部路径描述符生成 (这里使用全局平均池化简化)，参考的是SENet的结构
        b, c, h, w = x.size()
        y = F.adaptive_avg_pool2d(x, 1)
        y = self.conv_reduce(y)
        y = self.conv_expand(y)
        return identity * y + identity

# 3. Local Attention Module (LAM)
class LAM(BaseModule):
    def __init__(self, channels, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            ConvModule(channels, channels, 1, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU')),
            ConvModule(channels, channels, 1, act_cfg=dict(type='Sigmoid'), norm_cfg=None)
        )

    def forward(self, f_target, f_source):
        if f_source.shape[-2:] != f_target.shape[-2:]:
             f_source_aligned = F.interpolate(f_source, size=f_target.shape[-2:], mode='bilinear', align_corners=True)
        else:
             f_source_aligned = f_source
             
        attn = self.fc(self.avg_pool(f_source_aligned))
        return f_target + attn * f_source_aligned

class CrossPAFPN(BaseModule):
    """
    Strict implementation of Eq. (4) from PE-Transformer paper.
    Assumption:
        P3 = Stride 32 (Deep/High-level)
        P4 = Stride 16
        P5 = Stride 8 (Shallow/Low-level)
    """
    def __init__(self, in_channels, out_channels=256, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels

        # ======================================================
        # 1. 预处理层：生成 P3, P4, P5 初始特征
        # ======================================================
        # PEM 输出的是 f_l (Stride 8) 和 f_h (Stride 32)
        # 我们需要从中构建 P3, P4, P5
        
        # P3 对应 f_h (Stride 32)
        self.p3_conv = ConvModule(in_channels, out_channels, 1, norm_cfg=dict(type='BN'))
        
        # P5 对应 f_l (Stride 8)
        self.p5_conv = ConvModule(in_channels, out_channels, 1, norm_cfg=dict(type='BN'))
        
        # P4 (Stride 16) 需要从 P3 和 P5 生成
        # 这里设计一个融合层：Up(P3) + Down(P5) -> Conv
        self.p4_fusion = ConvModule(out_channels*2, out_channels, 3, padding=1, norm_cfg=dict(type='BN'))


        # ======================================================
        # 2. 跨路径聚合层 (Cross-Path Aggregation) - 对应 Eq.(4)
        # ======================================================
        
        # Branch P3' (Stride 32)
        # Input: Cat[P3, DS(P4)] -> 256 + 256 = 512 channels
        self.p3_prime_conv = ConvModule(
            out_channels * 2, out_channels, 3, padding=1, norm_cfg=dict(type='BN'))
        
        # Branch P4' (Stride 16)
        # Input: Cat[UP(P3), P4, UP(P3'), DS(P5)] -> 256*4 = 1024 channels
        self.p4_prime_conv = ConvModule(
            out_channels * 4, out_channels, 3, padding=1, norm_cfg=dict(type='BN'))
        
        # Branch P5' (Stride 8)
        # Input: Cat[UP(P4), P5, UP(P4')] -> 256*3 = 768 channels
        self.p5_prime_conv = ConvModule(
            out_channels * 3, out_channels, 3, padding=1, norm_cfg=dict(type='BN'))
        
        # Branch P6' (Stride 4 - Super Resolution)
        # Input: Cat[UP(P5), UP(P5')] -> 256*2 = 512 channels
        self.p6_prime_conv = ConvModule(
            out_channels * 2, out_channels, 3, padding=1, norm_cfg=dict(type='BN'))

        # Downsample module (Stride 2 conv)
        self.downsample = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, f_l_prime, f_h_prime):
        """
        f_l_prime: 来自 PEM 的浅层特征 (Stride 8) -> 对应论文 P5
        f_h_prime: 来自 PEM 的深层特征 (Stride 32) -> 对应论文 P3
        """
        
        # --- Step 1: 准备基础特征 P3, P4, P5 ---
        p3 = self.p3_conv(f_h_prime)  # S32
        p5 = self.p5_conv(f_l_prime)  # S8
        
        # 生成 P4 (S16)
        p3_up = F.interpolate(p3, scale_factor=2, mode='nearest')
        p5_down = self.downsample(p5)
        # 融合生成 P4
        p4 = self.p4_fusion(torch.cat([p3_up, p5_down], dim=1))
        
        # --- Step 2: 实现公式 (4) ---
        
        # 1. 计算 P3' (Stride 32)
        # f_{P3'}^i = Cat[f_p3; DS(f_p4)]
        p4_down = self.downsample(p4)
        p3_prime = self.p3_prime_conv(torch.cat([p3, p4_down], dim=1))
        
        # 2. 计算 P4' (Stride 16)
        # f_{P4'}^i = Cat[UP(f_p3); f_p4; f_p3'; DS(f_p5)]
        # 注意: 公式中的 f_p3' 需要上采样才能匹配 S16
        p3_up_layer = F.interpolate(p3, scale_factor=2, mode='nearest')
        p3_prime_up = F.interpolate(p3_prime, scale_factor=2, mode='nearest')
        p5_down_layer = self.downsample(p5)
        
        p4_prime = self.p4_prime_conv(torch.cat([
            p3_up_layer, 
            p4, 
            p3_prime_up, 
            p5_down_layer
        ], dim=1))
        
        # 3. 计算 P5' (Stride 8)
        # f_{P5'}^i = Cat[UP(f_p4); f_p5; f_p4']
        # 注意: f_p4 和 f_p4' 都需要上采样到 S8
        p4_up_layer = F.interpolate(p4, scale_factor=2, mode='nearest')
        p4_prime_up = F.interpolate(p4_prime, scale_factor=2, mode='nearest')
        
        p5_prime = self.p5_prime_conv(torch.cat([
            p4_up_layer,
            p5,
            p4_prime_up
        ], dim=1))
        
        # 4. 计算 P6' (Stride 4)
        # f_{P6'}^i = Cat[UP(f_p5); f_p5']
        # 注意: 这是为了检测极小目标生成的超高分辨率层
        p5_up_layer = F.interpolate(p5, scale_factor=2, mode='nearest')
        p5_prime_up = F.interpolate(p5_prime, scale_factor=2, mode='nearest')
        
        p6_prime = self.p6_prime_conv(torch.cat([
            p5_up_layer,
            p5_prime_up
        ], dim=1))
        
        # 返回结果 (按照 Stride 从小到大排序: S4, S8, S16, S32)
        # GroundingDINO 需要多尺度输入
        return (p6_prime, p5_prime, p4_prime, p3_prime)

# =================================================================
# 核心 Neck 模块：整合 PEM 和 Cross-PAFPN
# =================================================================
@MODELS.register_module()
class PETransformerNeck(BaseModule):
    """
    PE-Transformer Neck.
    Args:
        in_channels (list[int]): Backbone输出的通道数列表，如 [64, 128, 256, 512]
        out_channels (int): 输出特征图的通道数，默认 256
    """
    def __init__(self, in_channels, out_channels=256, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        
        # ============================================================
        # 1. 初始化 PEM 组件 (Feature Enhancer)
        # ============================================================
        # FAM: 融合相邻层特征
        # fam_low 输入: f0(S4), f1(S8) -> 输出 f_l(S4)
        self.fam_low = FAM(in_channels[0], in_channels[1], out_channels)
        # fam_high 输入: f2(S16), f3(S32) -> 输出 f_h(S16)
        self.fam_high = FAM(in_channels[2], in_channels[3], out_channels)
        
        # PAM: 路径注意力增强
        self.pam_low = PAM(out_channels)
        self.pam_high = PAM(out_channels)
        
        # LAM: 局部注意力互嵌入
        self.lam = LAM(out_channels)
        
        # ============================================================
        # 2. 初始化 特征对齐层 (Stride Alignment)
        # ============================================================
        # PEM 输出的是 S4 和 S16，但 CrossPAFPN (基于 FPN P3-P5 逻辑) 通常期望 S8 和 S32
        # 为了匹配论文中 P3/P5 的层级关系，我们在这里进行一次下采样对齐
        self.reduce_stride_l = ConvModule(
            out_channels, out_channels, 3, stride=2, padding=1, norm_cfg=dict(type='BN')) # S4 -> S8
        
        self.reduce_stride_h = ConvModule(
            out_channels, out_channels, 3, stride=2, padding=1, norm_cfg=dict(type='BN')) # S16 -> S32

        # ============================================================
        # 3. 初始化 Cross-PAFPN 组件 (Feature Fusion)
        # ============================================================
        # 使用你新写的 CrossPAFPN 类
        self.cross_pafpn = CrossPAFPN(out_channels, out_channels)

    def forward(self, inputs):
        # inputs 是来自 Backbone 的 tuple: (f0, f1, f2, f3)
        # f0: Stride 4, f1: Stride 8, f2: Stride 16, f3: Stride 32
        assert len(inputs) == 4
        f0, f1, f2, f3 = inputs
        
        # -------------------------------------------------------
        # Step 1: PEM 增强过程
        # -------------------------------------------------------
        # 1.1 FAM 聚合
        f_l = self.fam_low(f0, f1)   # Output: Stride 4
        f_h = self.fam_high(f2, f3)  # Output: Stride 16
        
        # 1.2 PAM 注意力过滤
        f_l = self.pam_low(f_l)
        f_h = self.pam_high(f_h)
        
        # 1.3 LAM 互嵌入 (特征增强的核心)
        f_h_prime = self.lam(f_target=f_h, f_source=f_l) 
        f_l_prime = self.lam(f_target=f_l, f_source=f_h)
        
        # -------------------------------------------------------
        # Step 2: 特征对齐 (适配 Cross-PAFPN 输入要求)
        # -------------------------------------------------------
        # 我们需要构造 P5 (浅层, S8) 和 P3 (深层, S32) 作为 Cross-PAFPN 的锚点
        
        # f_l_prime (S4) -> Downsample -> f_l_aligned (S8)
        f_l_aligned = self.reduce_stride_l(f_l_prime)
        
        # f_h_prime (S16) -> Downsample -> f_h_aligned (S32)
        f_h_aligned = self.reduce_stride_h(f_h_prime)
        
        # -------------------------------------------------------
        # Step 3: Cross-PAFPN 融合过程
        # -------------------------------------------------------
        # 输入: f_l_aligned (S8), f_h_aligned (S32)
        # 输出: Tuple (P6', P5', P4', P3') 对应 Strides (4, 8, 16, 32)
        outputs = self.cross_pafpn(f_l_aligned, f_h_aligned)
        
        # MMDetection Neck 输出通常要求是一个 tuple
        # 返回顺序: Stride 4, Stride 8, Stride 16, Stride 32
        return outputs