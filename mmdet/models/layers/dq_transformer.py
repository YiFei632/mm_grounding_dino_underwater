import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule, xavier_init
from mmdet.registry import MODELS

# ====================================================================
# 1. 辅助组件: CCM & CGFE
# ====================================================================

class CategoricalCounting(BaseModule):
    def __init__(self, cls_num=4, init_cfg=None):
        super(CategoricalCounting, self).__init__(init_cfg)
        self.ccm_cfg = [512, 512, 512, 256, 256, 256]
        self.in_channels = 512
        self.conv1 = nn.Conv2d(256, self.in_channels, kernel_size=1)
        self.ccm = self.make_layers(self.ccm_cfg, in_channels=self.in_channels, d_rate=2)
        self.output = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.linear = nn.Linear(256, cls_num)

    def make_layers(self, cfg, in_channels=3, batch_norm=False, d_rate=1):
        layers = []
        for v in cfg:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
        return nn.Sequential(*layers)

    def forward(self, features, spatial_shapes):
        bs, _, c = features.shape
        h, w = spatial_shapes[0]
        feat_len = int(h * w)
        v_feat = features[:, :feat_len, :].transpose(1, 2).view(bs, c, int(h), int(w))
        
        x = self.conv1(v_feat)
        density_map = self.ccm(x)
        out = self.output(density_map).flatten(1)
        cls_out = self.linear(out)
        return cls_out, density_map

class MultiScaleFeature(BaseModule):
    def __init__(self, channels=256, num_levels=5, init_cfg=None):
        super(MultiScaleFeature, self).__init__(init_cfg)
        self.convs = nn.ModuleList()
        for _ in range(num_levels - 1):
            self.convs.append(nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(32, channels)
            ))
    def forward(self, x):
        outs = [x]
        curr = x
        for conv in self.convs:
            curr = conv(curr)
            outs.append(curr)
        return outs

class ChannelGate(BaseModule):
    def __init__(self, gate_channels, reduction_ratio=16, init_cfg=None):
        super(ChannelGate, self).__init__(init_cfg)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
    def forward(self, x):
        avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att = self.mlp(avg_pool)
        scale = torch.sigmoid(channel_att).unsqueeze(2).unsqueeze(3).expand_as(x)
        return scale

class SpatialGate(BaseModule):
    def __init__(self, init_cfg=None):
        super(SpatialGate, self).__init__(init_cfg)
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1, eps=1e-5, momentum=0.01),
        )
    def forward(self, x):
        x_compress = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return scale

class CGFE(BaseModule):
    def __init__(self, gate_channels=256, num_feature_levels=5, init_cfg=None):
        super(CGFE, self).__init__(init_cfg)
        self.num_feat = num_feature_levels
        self.ChannelGate = ChannelGate(gate_channels)
        self.SpatialGate = SpatialGate()

    def forward(self, density_features, memory, spatial_shapes):
        bs, _, c = memory.shape
        enhanced_feats = []
        start_idx = 0
        memory_trans = memory.transpose(1, 2) 
        for i in range(self.num_feat):
            if i >= len(density_features): break
            h, w = spatial_shapes[i]
            end_idx = start_idx + int(h * w)
            feat = memory_trans[:, :, start_idx:end_idx].view(bs, c, int(h), int(w))
            
            sp_scale = self.SpatialGate(density_features[i])
            feat = feat * sp_scale
            ch_scale = self.ChannelGate(feat)
            feat = feat * ch_scale
            
            enhanced_feats.append(feat.flatten(2).transpose(1, 2))
            start_idx = end_idx
        return torch.cat(enhanced_feats, 1)

# ====================================================================
# 2. 核心: DQDeformableDetrTransformer
# ====================================================================

@MODELS.register_module()
class DQDeformableDetrTransformer(BaseModule):
    def __init__(self,
                 encoder=None,
                 decoder=None,
                 ccm_params=[10, 100, 500],
                 dynamic_query_list=[300, 500, 900, 1500],
                 ccm_cls_num=4,
                 embed_dims=256,
                 num_feature_levels=5,
                 init_cfg=None):
        super(DQDeformableDetrTransformer, self).__init__(init_cfg)
        
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.ccm_params = ccm_params
        self.dynamic_query_list = dynamic_query_list
        self.ccm_cls_num = ccm_cls_num

        if isinstance(encoder, nn.Module):
            self.encoder = encoder
        elif encoder is not None:
            self.encoder = MODELS.build(encoder)
            
        if isinstance(decoder, nn.Module):
            self.decoder = decoder
        elif decoder is not None:
            self.decoder = MODELS.build(decoder)
            
        self.CCM = CategoricalCounting(cls_num=ccm_cls_num)
        self.CGFE = CGFE(gate_channels=embed_dims, num_feature_levels=num_feature_levels)
        self.multiscale = MultiScaleFeature(channels=embed_dims, num_levels=num_feature_levels)

        self.level_embeds = nn.Parameter(torch.Tensor(self.num_feature_levels, self.embed_dims))
        
    def init_weights(self):
        super().init_weights()
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        xavier_init(self.level_embeds, distribution='uniform', bias=0)
        
    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N, S, C = memory.shape
        proposals = []
        _cur = 0
        for lvl, (H, W) in enumerate(spatial_shapes):
            H, W = int(H), int(W)
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H * W)]
            _cur += H * W
            
            mask_flatten_ = mask_flatten_.view(N, H, W)
            
            valid_H = torch.sum(~mask_flatten_[:, :, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :], 1)
            
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(0, H - 1, H, dtype=torch.float32, device=memory.device),
                torch.linspace(0, W - 1, W, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)
            
            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0**lvl)
            proposal = torch.cat((grid, wh), -1).view(N, -1, 4)
            proposals.append(proposal)
            
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))
        return output_proposals

    def forward(self,
                mlvl_feats,
                mlvl_masks,
                query_embed,
                mlvl_pos_embeds,
                reg_branches=None,
                cls_branches=None,
                **kwargs):
        
        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        
        for lvl, (feat, mask, pos_embed) in enumerate(zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            
            feat = feat.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
            
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in mlvl_masks], 1)

        memory = self.encoder(
            query=feat_flatten,
            query_pos=lvl_pos_embed_flatten,
            key_padding_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            **kwargs)

        ccm_logits, density_map = self.CCM(memory, spatial_shapes)
        
        _, predicted_cls = torch.max(ccm_logits.data, 1)
        max_cls_idx = predicted_cls.max().item()
        select_idx = min(max_cls_idx, len(self.dynamic_query_list)-1)
        num_select = self.dynamic_query_list[select_idx]
        
        multi_density_features = self.multiscale(density_map)
        memory_enhanced = self.CGFE(multi_density_features, memory, spatial_shapes)

        enc_outputs_class = cls_branches[self.decoder.num_layers](memory_enhanced)
        enc_outputs_coord_unact = reg_branches[self.decoder.num_layers](memory_enhanced) + \
                                  self.gen_encoder_output_proposals(memory_enhanced, mask_flatten, spatial_shapes)

        topk = min(num_select, enc_outputs_class.shape[1])
        topk_proposals = torch.topk(enc_outputs_class.max(-1)[0], topk, dim=1)[1]
        
        topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
        topk_coords_unact = topk_coords_unact.detach()
        reference_points = topk_coords_unact.sigmoid()
        init_reference_out = reference_points
        
        pos_trans_out = torch.gather(memory_enhanced, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, self.embed_dims)).detach()
        query = pos_trans_out

        # -----------------------------------------------------------
        # Step 4: Decoder
        # -----------------------------------------------------------
        inter_states, inter_references = self.decoder(
            query=query,
            query_pos=None,  # 【关键修正】显式传递 query_pos=None
            value=memory_enhanced,
            key_padding_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=reg_branches,
            **kwargs)

        return inter_states, init_reference_out, inter_references, \
               enc_outputs_class, enc_outputs_coord_unact.sigmoid(), ccm_logits