import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.ops import MultiScaleDeformableAttention
from mmdet.registry import MODELS
from mmengine.model import BaseModule, ModuleList

class CSPRepLayer(BaseModule):
    """简化版的 CSP Layer，适配 MMDet 风格"""
    def __init__(self, in_channels, out_channels, num_blocks=3, expansion=1.0, bias=False, act_cfg=dict(type='SiLU')):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvModule(in_channels, hidden_channels, 1, 1, bias=bias, act_cfg=act_cfg)
        self.conv2 = ConvModule(in_channels, hidden_channels, 1, 1, bias=bias, act_cfg=act_cfg)
        self.bottlenecks = nn.Sequential(*[
            ConvModule(hidden_channels, hidden_channels, 3, 1, padding=1, bias=False, act_cfg=act_cfg) 
            for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvModule(hidden_channels, out_channels, 1, 1, bias=bias, act_cfg=act_cfg)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)

@MODELS.register_module()
class UCFHybridEncoder(BaseModule):
    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 num_levels=3,
                 num_points=4,
                 dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.feat_strides = feat_strides
        
        # 1. Input Projection
        self.input_proj = ModuleList([
            nn.Sequential(
                nn.Conv2d(in_chan, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim)
            ) for in_chan in in_channels
        ])

        # 2. Encoder (AIFI)
        self.encoder_layer = nn.TransformerEncoderLayer(hidden_dim, nhead, dim_feedforward=1024, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

        # 3. Bi-directional Cross-Attention Layers (CFP Core)
        # Raw stream uses this to attend to Enhanced stream
        self.cross_attn_raw = ModuleList([
            MultiScaleDeformableAttention(embed_dims=hidden_dim, num_levels=1, num_points=num_points, batch_first=True)
            for _ in range(len(in_channels) - 1)
        ])
        # Enhanced stream uses this to attend to Raw stream
        self.cross_attn_enh = ModuleList([
            MultiScaleDeformableAttention(embed_dims=hidden_dim, num_levels=1, num_points=num_points, batch_first=True)
            for _ in range(len(in_channels) - 1)
        ])
        
        self.dropouts = ModuleList([nn.Dropout(dropout) for _ in range(len(in_channels) - 1)])
        self.norms_raw = ModuleList([nn.LayerNorm(hidden_dim) for _ in range(len(in_channels) - 1)])
        self.norms_enh = ModuleList([nn.LayerNorm(hidden_dim) for _ in range(len(in_channels) - 1)])

        # 4. FPN & PAN Blocks
        self.lateral_convs = ModuleList()
        self.fpn_blocks = ModuleList()
        self.downsample_convs = ModuleList()
        self.pan_blocks = ModuleList()

        for _ in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(ConvModule(hidden_dim, hidden_dim, 1, 1, act_cfg=dict(type='SiLU')))
            self.fpn_blocks.append(CSPRepLayer(hidden_dim * 2, hidden_dim))

        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(ConvModule(hidden_dim, hidden_dim, 3, 2, padding=1, act_cfg=dict(type='SiLU')))
            self.pan_blocks.append(CSPRepLayer(hidden_dim * 2, hidden_dim))

    def get_reference_points(self, h, w, device):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, h - 0.5, h, dtype=torch.float32, device=device),
            torch.linspace(0.5, w - 0.5, w, dtype=torch.float32, device=device),
            indexing='ij')
        ref = torch.stack((ref_x, ref_y), -1)  # [H, W, 2] (x, y)
        return ref.reshape(-1, 1, 2) / torch.tensor([w, h], device=device) # Normalize

    def forward(self, feats_raw, feats_enh):
        # Project features
        proj_raw = [self.input_proj[i](feat) for i, feat in enumerate(feats_raw)]
        proj_enh = [self.input_proj[i](feat) for i, feat in enumerate(feats_enh)]
        
        # ---------------- AIFI (Encoder) on High-level features ----------------
        def run_encoder(feats):
            feat_high = feats[-1] # [B, C, H, W]
            bs, c, h, w = feat_high.shape
            src = feat_high.flatten(2).permute(0, 2, 1) # [B, HW, C]
            memory = self.encoder(src)
            return memory.permute(0, 2, 1).reshape(bs, c, h, w)

        proj_raw[-1] = run_encoder(proj_raw)
        proj_enh[-1] = run_encoder(proj_enh)

        # ---------------- Top-down FPN ----------------
        def run_fpn(feats):
            inner_outs = [feats[-1]]
            for idx in range(len(self.in_channels) - 1, 0, -1):
                feat_high = inner_outs[0]
                feat_low = feats[idx - 1]
                
                feat_high = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_high)
                inner_outs[0] = feat_high
                
                upsample_feat = F.interpolate(feat_high, scale_factor=2., mode='nearest')
                inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](torch.cat([upsample_feat, feat_low], dim=1))
                inner_outs.insert(0, inner_out)
            return inner_outs

        fpn_raw = run_fpn(proj_raw)
        fpn_enh = run_fpn(proj_enh)

        # ---------------- Bottom-up PAN with CFP ----------------
        outs_raw = [fpn_raw[0]]
        outs_enh = [fpn_enh[0]]
        
        for idx in range(len(self.in_channels) - 1):
            # idx maps to level [0 -> 1], [1 -> 2]
            
            # Current level (already fused)
            curr_raw = outs_raw[-1]
            curr_enh = outs_enh[-1]
            
            # Next level (from FPN)
            next_raw = fpn_raw[idx + 1]
            next_enh = fpn_enh[idx + 1]
            
            bs, c, h, w = next_raw.shape
            
            # 关键修改：强制 contiguous() 并在 Permute 后确保内存布局清晰
            # Query: [B, L, C]
            query_raw = next_raw.flatten(2).permute(0, 2, 1).contiguous()
            query_enh = next_enh.flatten(2).permute(0, 2, 1).contiguous()
            
            ref_points = self.get_reference_points(h, w, next_raw.device).expand(bs, -1, -1, 2) # [B, L, 1, 2]
            
            # 关键修改：强制 dtype=torch.long
            spatial_shapes = torch.as_tensor([(h, w)], device=next_raw.device, dtype=torch.long)
            level_start_index = torch.tensor([0], device=next_raw.device, dtype=torch.long)

            # Cross Attention: Raw Query <-> Enh Value
            raw_cross = self.cross_attn_raw[idx](
                query=query_raw, 
                value=query_enh, 
                reference_points=ref_points, 
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index)
            
            fused_raw_feat = query_raw + self.dropouts[idx](raw_cross)
            fused_raw_feat = self.norms_raw[idx](fused_raw_feat).permute(0, 2, 1).reshape(bs, c, h, w)

            # Cross Attention: Enh Query <-> Raw Value
            enh_cross = self.cross_attn_enh[idx](
                query=query_enh,
                value=query_raw,
                reference_points=ref_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index)
            
            fused_enh_feat = query_enh + self.dropouts[idx](enh_cross)
            fused_enh_feat = self.norms_enh[idx](fused_enh_feat).permute(0, 2, 1).reshape(bs, c, h, w)
            
            # PAN Fusion
            def run_pan_block(prev_feat, target_feat, block):
                down = self.downsample_convs[idx](prev_feat)
                return block(torch.cat([down, target_feat], dim=1))

            out_raw = run_pan_block(curr_raw, fused_raw_feat, self.pan_blocks[idx])
            out_enh = run_pan_block(curr_enh, fused_enh_feat, self.pan_blocks[idx])
            
            outs_raw.append(out_raw)
            outs_enh.append(out_enh)
            
        return outs_raw, outs_enh