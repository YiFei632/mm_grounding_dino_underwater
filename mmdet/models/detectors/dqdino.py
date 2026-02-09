# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import Tensor, nn
from torch.nn.init import normal_
from typing import Dict, Optional, Tuple, List

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList
from mmdet.utils import OptConfigType
from ..layers import (CdnQueryGenerator, DeformableDetrTransformerEncoder,
                      DinoTransformerDecoder, SinePositionalEncoding)
from .deformable_detr import DeformableDETR, MultiScaleDeformableAttention

# 假设你已经将 ccm.py 和 cgfe.py 放在了同级目录的 modules 文件夹下
# 如果放在同级目录，请改为 from .ccm import CategoricalCounting 等
from mmdet.models.utils import CategoricalCounting
from mmdet.models.utils import CGFE, MultiScaleFeature

@MODELS.register_module()
class DQDINO(DeformableDETR):
    """
    DQ-DETR Transformer implementation.
    Integrates CCM (Categorical Counting Module) and CGFE (Context-Guided Feature Enhancement).
    """

    def __init__(self,
                 *args,
                 ccm_cfg: OptConfigType = None,
                 cgfe_cfg: OptConfigType = None,
                 dynamic_query_list = None,
                 dn_cfg: OptConfigType = None,
                 **kwargs) -> None:

        # print(kwargs.keys())
        # print(f"1. ccm_cfg: {ccm_cfg}")             # <--- 直接打印变量名
        # print(f"2. cgfe_cfg: {cgfe_cfg}")           # <--- 直接打印变量名
        # print(f"3. dynamic_query_list: {dynamic_query_list}")
        super().__init__(*args, **kwargs)
        
        assert self.as_two_stage, 'as_two_stage must be True for DINO'
        assert self.with_box_refine, 'with_box_refine must be True for DINO'
        
        # 1. 初始化 DQ-DETR 特有模块
        # 注意：这里需要确保 ccm_cfg 和 cgfe_cfg 在 config 中已定义
        self.ccm = CategoricalCounting(**(ccm_cfg or {}))
        self.cgfe = CGFE(**(cgfe_cfg or {}))
        
        # DQ-DETR 默认使用 5 尺度的 MultiScaleFeature，但 MMDet DINO 默认是 4 尺度
        # 需要确保这里的 num_levels 与 backbone/neck 输出一致
        is_5_scale = self.num_feature_levels == 5
        self.multiscale = MultiScaleFeature(is_5_scale=is_5_scale)

        if dn_cfg is not None:
            assert 'num_classes' not in dn_cfg and \
                   'num_queries' not in dn_cfg and \
                   'hidden_dim' not in dn_cfg, \
                'The three keyword args `num_classes`, `embed_dims`, and ' \
                '`num_matching_queries` are set in `detector.__init__()`, ' \
                'users should not set them in `dn_cfg` config.'
            dn_cfg['num_classes'] = self.bbox_head.num_classes
            dn_cfg['embed_dims'] = self.embed_dims
            dn_cfg['num_matching_queries'] = self.num_queries
        
        self.dynamic_query_list = dynamic_query_list
        
        # 记录当前的 query 数量，用于日志或调试
        self.current_num_queries = self.num_queries
        self.dn_query_generator = CdnQueryGenerator(**dn_cfg)

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        self.encoder = DeformableDetrTransformerEncoder(**self.encoder)
        self.decoder = DinoTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)
        # NOTE In DINO, the query_embedding only contains content
        # queries, while in Deformable DETR, the query_embedding
        # contains both content and spatial queries, and in DETR,
        # it only contains spatial queries.

        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            f'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'

        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)

    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super(DeformableDETR, self).init_weights()
        for coder in self.encoder, self.decoder:
            for p in coder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        nn.init.xavier_uniform_(self.memory_trans_fc.weight)
        nn.init.xavier_uniform_(self.query_embedding.weight)
        normal_(self.level_embed)
    
    def forward_transformer(self,
        img_feats: Tuple[Tensor],
        batch_data_samples: OptSampleList = None,
    ) -> Dict:
        """
        Overriding forward to insert CCM and CGFE logic between Encoder and Decoder.
        """
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)
        # -----------------------------------------------------------
        # Step 1: Encoder
        # -----------------------------------------------------------
        encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict)
            # 注意：encoder 返回值可能因版本略有不同，通常是 (memory, spatial_shapes, ...)
        # 开始加入修改逻辑
        # print(encoder_outputs_dict.keys())
        memory = encoder_outputs_dict['memory']
        memory_mask = encoder_outputs_dict['memory_mask']
        spatial_shapes = encoder_outputs_dict['spatial_shapes']
        

        # -----------------------------------------------------------
        # Step 2: DQ-DETR Core Logic (CCM & CGFE)
        # -----------------------------------------------------------
        
        # 2.1 CCM 需要恢复空间结构
        # DQ-DETR 的 CCM 实现依赖于特定的空间结构输入，这里需要适配
        # 假设 memory 是 (bs, num_feat, c)
        
        # 运行 CCM
        counting_output, ccm_feature = self.ccm(memory, spatial_shapes)
        
        # 2.2 CGFE 特征增强
        multi_ccm_feature = self.multiscale(ccm_feature)
        
        # memory_enhanced: (bs, num_feat, c)
        memory_enhanced = self.cgfe(multi_ccm_feature, memory, spatial_shapes)

        if memory_mask is not None:
            # 检查特征长度和掩码长度是否一致
            feat_len = memory_enhanced.size(1)
            mask_len = memory_mask.size(1)
            
            if feat_len > mask_len:
                # 计算缺失的长度 (例如 224)
                diff = feat_len - mask_len
                
                # 创建全为 False (Valid) 的掩码
                # 注意：在 MMDetection Transformer 中，False 表示有效像素，True 表示 Padding
                extra_mask = torch.zeros(
                    (memory_mask.size(0), diff),
                    dtype=memory_mask.dtype,
                    device=memory_mask.device
                )
                
                # 拼接到原 mask 后面
                memory_mask = torch.cat([memory_mask, extra_mask], dim=1)

        # 2.3 动态选择 Query 数量
        # 获取分类结果中概率最大的类别索引
        # counting_output: (bs, num_classes)
        # Use detach() only for index selection to avoid gradient issues
        _, predicted_density_idx = torch.max(counting_output, 1)
        
        # 策略：取当前 Batch 中最拥挤的那张图对应的等级，作为整个 Batch 的 Query 数量
        # 这样可以保持 Batch Tensor 维度对齐
        if self.training:
             # 训练时也可以选择使用最大值，或者固定使用最大配置以稳定训练
            batch_max_idx = max(predicted_density_idx.tolist())
            num_select = self.dynamic_query_list[batch_max_idx]
        else:
            # 推理时动态调整
            batch_max_idx = max(predicted_density_idx.tolist())
            num_select = self.dynamic_query_list[batch_max_idx]

        self.current_num_queries = num_select

        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            memory_enhanced, memory_mask, num_select, spatial_shapes ,batch_data_samples=batch_data_samples)
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)

        # Ensure CCM parameters get gradients in DDP training
        # Add counting_output with a very small coefficient to prevent "Expected to mark
        # a variable ready only once" error in distributed training
        if self.training:
            # Add a negligible term to ensure all CCM parameters participate in gradient computation
            # This is critical for DDP training to work correctly
            head_inputs_dict['hidden_states'][0] = head_inputs_dict['hidden_states'][0] + \
                counting_output.sum() * 1e-10

        return head_inputs_dict


    

    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        num_select,
        spatial_shapes: Tensor,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict, Dict]:



        bs, _, c = memory.shape
        cls_out_features = self.bbox_head.cls_branches[
            self.decoder.num_layers].out_features

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)
        enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers](
                output_memory)
        enc_outputs_coord_unact = self.bbox_head.reg_branches[
            self.decoder.num_layers](output_memory) + output_proposals

        # NOTE The DINO selects top-k proposals according to scores of
        # multi-class classification, while DeformDETR, where the input
        # is `enc_outputs_class[..., 0]` selects according to scores of
        # binary classification.
        topk_indices = torch.topk(
            enc_outputs_class.max(-1)[0], k=num_select, dim=1)[1]
        topk_score = torch.gather(
            enc_outputs_class, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        topk_coords = topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()

        query = self.query_embedding.weight[:num_select, None, :]
        query = query.repeat(1, bs, 1).transpose(0, 1)
        if self.training:
            # Temporarily modify num_matching_queries to match dynamic num_select
            original_num_matching_queries = self.dn_query_generator.num_matching_queries
            self.dn_query_generator.num_matching_queries = num_select

            dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
                self.dn_query_generator(batch_data_samples)

            # Restore original value
            self.dn_query_generator.num_matching_queries = original_num_matching_queries

            query = torch.cat([dn_label_query, query], dim=1)
            reference_points = torch.cat([dn_bbox_query, topk_coords_unact],
                                         dim=1)
        else:
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None
        reference_points = reference_points.sigmoid()

        decoder_inputs_dict = dict(
            query=query,
            memory=memory,
            reference_points=reference_points,
            dn_mask=dn_mask)
        # NOTE DINO calculates encoder losses on scores and coordinates
        # of selected top-k encoder queries, while DeformDETR is of all
        # encoder queries.
        head_inputs_dict = dict(
            enc_outputs_class=topk_score,
            enc_outputs_coord=topk_coords,
            dn_meta=dn_meta) if self.training else dict()
        return decoder_inputs_dict, head_inputs_dict
    
    def forward_decoder(self,
                        query: Tensor,
                        memory: Tensor,
                        memory_mask: Tensor,
                        reference_points: Tensor,
                        spatial_shapes: Tensor,
                        level_start_index: Tensor,
                        valid_ratios: Tensor,
                        dn_mask: Optional[Tensor] = None,
                        **kwargs) -> Dict:
        """Forward with Transformer decoder.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            query (Tensor): The queries of decoder inputs, has shape
                (bs, num_queries_total, dim), where `num_queries_total` is the
                sum of `num_denoising_queries` and `num_matching_queries` when
                `self.training` is `True`, else `num_matching_queries`.
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat_points).
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries_total, 4) with the last dimension arranged as
                (cx, cy, w, h).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            dn_mask (Tensor, optional): The attention mask to prevent
                information leakage from different denoising groups and
                matching parts, will be used as `self_attn_mask` of the
                `self.decoder`, has shape (num_queries_total,
                num_queries_total).
                It is `None` when `self.training` is `False`.

        Returns:
            dict: The dictionary of decoder outputs, which includes the
            `hidden_states` of the decoder output and `references` including
            the initial and intermediate reference_points.
        """
        inter_states, references = self.decoder(
            query=query,
            value=memory,
            key_padding_mask=memory_mask,
            self_attn_mask=dn_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=self.bbox_head.reg_branches,
            **kwargs)

        if self.training:
            # NOTE: This is to make sure label_embeding can be involved to
            # produce loss even if there is no denoising query (no ground truth
            # target in this GPU), otherwise, this will raise runtime error in
            # distributed training.
            # In DQDINO, we always add this term to ensure the parameter gets gradients
            inter_states[0] = inter_states[0] + \
                self.dn_query_generator.label_embedding.weight[0, 0] * 0.0

        decoder_outputs_dict = dict(
            hidden_states=inter_states, references=list(references))
        return decoder_outputs_dict