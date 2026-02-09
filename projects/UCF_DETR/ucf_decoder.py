import torch
import torch.nn as nn
from mmengine.model import BaseModule, ModuleList
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.ops import MultiScaleDeformableAttention
from mmdet.registry import MODELS

# -------------------------------------------------------------------------
# 1. CQI: Cross-domain Query Interaction Module
# -------------------------------------------------------------------------
class QueryInteraction(BaseModule):
    """CQI: Cross-domain Query Interaction"""
    def __init__(self, d_model, n_head=8, dropout=0., adapter_ratio=4):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(d_model, d_model // adapter_ratio),
            nn.ReLU(),
            nn.Linear(d_model // adapter_ratio, d_model),
            nn.LayerNorm(d_model)
        )
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Gating network
        self.gate_net = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        nn.init.constant_(self.gate_net[-2].bias, -4.0)

    def forward(self, tgt, ref, pos_tgt=None, pos_ref=None, attn_mask=None):
        ref_adapted = self.adapter(ref)
        
        q = tgt + pos_tgt if pos_tgt is not None else tgt
        k = ref_adapted + pos_ref if pos_ref is not None else ref_adapted
        v = ref_adapted
        
        tgt2, _ = self.attn(query=q, key=k, value=v, attn_mask=attn_mask)
        tgt2 = self.dropout(tgt2)
        
        gate = self.gate_net(torch.cat([tgt, tgt2], dim=-1))
        tgt = self.norm(tgt + gate * tgt2)
        return tgt

# -------------------------------------------------------------------------
# 2. Layer: UCFDetrTransformerDecoderLayer
# -------------------------------------------------------------------------
@MODELS.register_module()
class UCFDetrTransformerDecoderLayer(BaseModule):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.,
                 self_attn_cfg=None,
                 cross_attn_cfg=None,
                 ffn_cfg=None,
                 **kwargs):
        super().__init__()
        
        # 1. Self Attention
        if self_attn_cfg is None:
            self_attn_cfg = dict(embed_dims=d_model, num_heads=n_head, dropout=dropout)
        self_attn_cfg['batch_first'] = True
        self.self_attn = MultiheadAttention(**self_attn_cfg)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # 2. Cross Attention (Deformable)
        if cross_attn_cfg is None:
            cross_attn_cfg = dict(embed_dims=d_model, num_levels=3)
        
        # 强制开启 batch_first
        cross_attn_cfg['batch_first'] = True
        
        self.cross_attn = MultiScaleDeformableAttention(**cross_attn_cfg)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        # 3. CQI Modules
        self.query_interaction = QueryInteraction(d_model, n_head, dropout)
        self.query_interaction_inv = QueryInteraction(d_model, n_head, dropout)

        # 4. FFN
        if ffn_cfg is None:
            ffn_cfg = dict(
                embed_dims=d_model,
                feedforward_channels=dim_feedforward,
                num_fcs=2,
                ffn_drop=dropout,
                act_cfg=dict(type='ReLU', inplace=True))
        self.ffn = FFN(**ffn_cfg)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, 
                query, 
                key=None, 
                value=None, 
                query_pos=None, 
                key_pos=None, 
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                valid_ratios=None,
                **kwargs):
        
        # --- Self Attention ---
        q = query + query_pos if query_pos is not None else query
        k = q 
        target2 = self.self_attn(query=q, key=k, value=query)
        query = self.norm1(query + self.dropout1(target2))
        
        # --- Cross Attention ---
        q = query + query_pos if query_pos is not None else query
        
        target2 = self.cross_attn(
            query=q,
            key=key, 
            value=value, 
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios
        )
        query = self.norm2(query + self.dropout2(target2))
        
        # --- CQI (Cross-domain Interaction) ---
        bs = query.size(0)
        if bs % 2 == 0 and bs > 0:
            q_raw, q_enh = query.split(bs // 2, dim=0)
            
            pos_raw, pos_enh = None, None
            if query_pos is not None:
                pos_raw, pos_enh = query_pos.split(bs // 2, dim=0)
            
            q_raw_new = self.query_interaction(q_raw, q_enh, pos_raw, pos_enh)
            q_enh_new = self.query_interaction_inv(q_enh, q_raw, pos_enh, pos_raw)
            
            query = torch.cat([q_raw_new, q_enh_new], dim=0)
        
        # --- FFN ---
        target2 = self.ffn(query)
        query = self.norm3(query + target2)
        
        return query

# -------------------------------------------------------------------------
# 3. Container: UCFTransformerDecoder
# -------------------------------------------------------------------------
@MODELS.register_module()
class UCFTransformerDecoder(BaseModule):
    def __init__(self, num_layers, layer_cfg, return_intermediate=True, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.layers = ModuleList()
        for _ in range(num_layers):
            self.layers.append(MODELS.build(layer_cfg))
            
    def forward(self, query, reference_points, feat, batch_data_samples=None, **kwargs):
        intermediate = []
        
        # 1. 准备 Metadata
        spatial_shapes = []
        for f in feat:
            spatial_shapes.append(f.shape[-2:])
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=query.device)
        
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        
        # 2. Flatten Memory
        memory_list = []
        for f in feat:
            memory_list.append(f.flatten(2).transpose(1, 2))
        memory = torch.cat(memory_list, 1).contiguous() 
        
        # 3. 准备 valid_ratios
        valid_ratios = None
        if batch_data_samples is not None:
            ratios_list = [self.get_valid_ratio(m, len(feat)) for m in batch_data_samples]
            valid_ratios = torch.stack(ratios_list, dim=0).to(query.device)
            
            if query.size(0) == 2 * valid_ratios.size(0):
                valid_ratios = torch.cat([valid_ratios, valid_ratios], dim=0)

        # === 关键修复：扩展 reference_points 维度 ===
        # MMCV MultiScaleDeformableAttention 期望: [B, N, NumLevels, 4]
        # 当前输入: [B, N, 4]
        if reference_points.dim() == 3:
            num_levels = len(feat)
            reference_points = reference_points.unsqueeze(2).repeat(1, 1, num_levels, 1)

        output = query
        for layer in self.layers:
            output = layer(
                query=output,
                value=memory,
                query_pos=None, 
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios
            )
            
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)
        
        return output.unsqueeze(0)

    def get_valid_ratio(self, batch_data_sample, num_levels):
        img_meta = batch_data_sample.metainfo
        valid_h, valid_w = img_meta.get('img_shape', (1, 1))[:2]
        pad_h, pad_w = img_meta.get('pad_shape', (1, 1))[:2]
        
        ratio_w = valid_w / pad_w
        ratio_h = valid_h / pad_h
        
        return torch.tensor([ratio_w, ratio_h]).unsqueeze(0).repeat(num_levels, 1)