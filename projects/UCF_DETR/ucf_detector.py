import torch
import torch.nn as nn
from mmdet.registry import MODELS
from mmdet.models.detectors import SingleStageDetector
from mmdet.models.dense_heads import DETRHead
from mmdet.structures import OptSampleList
from typing import Dict, List, Optional, Tuple, Union
from torch import Tensor
from mmengine.structures import InstanceData
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy
from .ucf_enhancer import MSREnhancement

@MODELS.register_module()
class UCFRTDETRHead(DETRHead):
    """适配 RT-DETR 的 Head，包含 Query Selection"""
    def __init__(self, 
                 num_classes, 
                 in_channels, 
                 transformer=None,
                 num_query=300,
                 **kwargs):
        
        self.in_channels = in_channels
        self.num_query = num_query
        
        super_kwargs = kwargs.copy()
        if 'embed_dims' not in super_kwargs:
            super_kwargs['embed_dims'] = in_channels
        keys_to_remove = ['transformer', 'in_channels', 'num_query', 'num_queries']
        for k in keys_to_remove:
            super_kwargs.pop(k, None)
            
        try:
            super().__init__(num_classes=num_classes, **super_kwargs)
        except TypeError:
            minimal_kwargs = {k: v for k, v in super_kwargs.items() 
                              if k in ['loss_cls', 'loss_bbox', 'loss_iou', 'train_cfg', 'test_cfg']}
            super().__init__(num_classes=num_classes, **minimal_kwargs)

        if transformer is not None:
            self.transformer = MODELS.build(transformer)
        elif not hasattr(self, 'transformer'):
            raise ValueError("Transformer configuration must be provided")
            
        self.query_embedding = None 
        self.enc_output = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.LayerNorm(in_channels)
        )
        self.enc_score_head = nn.Linear(in_channels, num_classes)
        self.enc_bbox_head = nn.Linear(in_channels, 4)

    def forward(self, x, batch_data_samples):
        # 1. Query Selection
        memory = x[-1] 
        bs, c, h, w = memory.shape
        memory_flat = memory.flatten(2).permute(0, 2, 1) 
        
        enc_out = self.enc_output(memory_flat)
        enc_scores = self.enc_score_head(enc_out)
        enc_bboxes = self.enc_bbox_head(enc_out).sigmoid()
        
        # 动态 Top-K：防止特征图过小
        valid_points = enc_scores.shape[1]
        topk = min(self.num_query, valid_points)
        
        topk_scores, topk_ind = torch.topk(enc_scores.max(-1).values, topk, dim=1)
        
        batch_ind = torch.arange(bs, device=memory.device).unsqueeze(1)
        init_content = enc_out[batch_ind, topk_ind]
        init_ref_points = enc_bboxes[batch_ind, topk_ind]

        # 2. Decoder
        decoder_inputs_dict = dict(
            query=init_content,
            reference_points=init_ref_points,
            feat=x, 
            batch_data_samples=batch_data_samples
        )
        
        dec_hidden_states = self.transformer(**decoder_inputs_dict)
        
        # 3. Predict Heads
        outputs_classes = []
        outputs_coords = []
        
        for layer_idx in range(dec_hidden_states.shape[0]):
            layer_hidden = dec_hidden_states[layer_idx]
            cls_score = self.fc_cls(layer_hidden)
            
            ref_points_sigmoid = init_ref_points 
            delta_bbox = self.fc_reg(layer_hidden)
            
            epsilon = 1e-5
            ref_points_sigmoid = torch.clamp(ref_points_sigmoid, epsilon, 1 - epsilon)
            ref_points_logit = torch.log(ref_points_sigmoid) - torch.log(1 - ref_points_sigmoid)
            
            bbox_pred = (ref_points_logit + delta_bbox).sigmoid()
            
            outputs_classes.append(cls_score)
            outputs_coords.append(bbox_pred)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        
        return dict(
            pred_logits=outputs_classes[-1],        
            pred_bboxes=outputs_coords[-1],         
            all_layers_cls_scores=outputs_classes,  
            all_layers_bbox_preds=outputs_coords,   
            enc_outputs_class=enc_scores,           
            enc_outputs_coord=enc_bboxes            
        )

    def predict(self, x, batch_data_samples, rescale=False):
        """
        手动实现后处理，避免调用不稳定的 predict_by_feat
        """
        outs = self.forward(x, batch_data_samples)
        
        pred_logits = outs['pred_logits'] # [B, N, NumClasses]
        pred_bboxes = outs['pred_bboxes'] # [B, N, 4] (cx, cy, w, h)
        
        # 只取 Raw 分支
        B_real = len(batch_data_samples)
        if pred_logits.size(0) == 2 * B_real:
            pred_logits = pred_logits[:B_real]
            pred_bboxes = pred_bboxes[:B_real]

        batch_img_metas = [data_samples.metainfo for data_samples in batch_data_samples]
        result_list = []

        # 获取配置中的 max_per_img
        cfg_max_per_img = self.test_cfg.get('max_per_img', 300)

        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            cls_score = pred_logits[img_id]
            bbox_pred = pred_bboxes[img_id]
            
            # 1. Sigmoid
            cls_score = cls_score.sigmoid()
            
            # 2. Flatten [N, NumClasses] -> [N * NumClasses]
            scores = cls_score.flatten()
            
            # 3. Dynamic Top-K (修复 Crash 的核心)
            # 确保 k 不超过总预测数量
            num_predictions = scores.numel()
            k = min(cfg_max_per_img, num_predictions)
            
            topk_values, topk_indexes = scores.topk(k)
            
            # 4. Decode Top-K
            # topk_indexes 是 flatten 后的索引
            # 对应的 anchor 索引: div
            # 对应的 class 索引: mod
            topk_bbox_indices = topk_indexes // self.num_classes
            topk_labels = topk_indexes % self.num_classes
            
            topk_bboxes = bbox_pred[topk_bbox_indices]
            
            # 5. BBox Conversion (cxcywh -> xyxy)
            # 输出是归一化的，需要根据 img_shape 还原
            h, w = img_meta['img_shape'][:2]
            topk_bboxes = bbox_cxcywh_to_xyxy(topk_bboxes)
            topk_bboxes[:, 0::2] *= w
            topk_bboxes[:, 1::2] *= h
            
            # 如果需要 rescale (还原到原始输入图片的尺寸)
            if rescale:
                assert img_meta.get('scale_factor') is not None
                scale_factor = topk_bboxes.new_tensor(img_meta['scale_factor']).repeat((1, 2))
                topk_bboxes /= scale_factor

            # 6. Pack Results
            results = InstanceData()
            results.bboxes = topk_bboxes
            results.scores = topk_values
            results.labels = topk_labels
            
            result_list.append(results)

        return result_list

    def loss(self, x: Tuple[Tensor], batch_data_samples: OptSampleList) -> dict:
        B_feat = x[0].size(0)
        B_gt = len(batch_data_samples)
        
        if B_feat == 2 * B_gt:
            batch_data_samples_expanded = batch_data_samples + batch_data_samples
        else:
            batch_data_samples_expanded = batch_data_samples

        outputs = self.forward(x, batch_data_samples_expanded)
        
        losses = self.loss_by_feat(
            all_layers_cls_scores=outputs['all_layers_cls_scores'],
            all_layers_bbox_preds=outputs['all_layers_bbox_preds'],
            batch_gt_instances=[data_sample.gt_instances for data_sample in batch_data_samples_expanded],
            batch_img_metas=[data_sample.metainfo for data_sample in batch_data_samples_expanded],
            batch_gt_instances_ignore=None
        )

        enc_cls = outputs['enc_outputs_class'].unsqueeze(0)
        enc_bbox = outputs['enc_outputs_coord'].unsqueeze(0)
        
        losses_enc = self.loss_by_feat(
            all_layers_cls_scores=enc_cls,
            all_layers_bbox_preds=enc_bbox,
            batch_gt_instances=[data_sample.gt_instances for data_sample in batch_data_samples_expanded],
            batch_img_metas=[data_sample.metainfo for data_sample in batch_data_samples_expanded],
            batch_gt_instances_ignore=None
        )
        
        for k, v in losses_enc.items():
            losses[f'{k}_enc'] = v
            
        return losses

@MODELS.register_module()
class UCFRTDETR(SingleStageDetector):
    def __init__(self, enhancer_cfg, bbox_head, neck, backbone, **kwargs):
        super().__init__(backbone=backbone, neck=neck, bbox_head=bbox_head, **kwargs)
        self.enhancer = MSREnhancement(**enhancer_cfg)

    def extract_feat(self, batch_inputs):
        if hasattr(self, 'data_preprocessor') and hasattr(self.data_preprocessor, 'mean'):
            mean = self.data_preprocessor.mean
            std = self.data_preprocessor.std
        else:
            device = batch_inputs.device
            mean = torch.tensor([123.675, 116.28, 103.53], device=device).view(1, 3, 1, 1)
            std = torch.tensor([58.395, 57.12, 57.375], device=device).view(1, 3, 1, 1)

        img_unnorm = batch_inputs * std + mean
        img_01 = img_unnorm / 255.0
        img_01 = torch.clamp(img_01, 0.0, 1.0) 
        
        with torch.no_grad():
            enhanced_01 = self.enhancer(img_01)
        
        enhanced_inputs = (enhanced_01 * 255.0 - mean) / std
        
        combined_inputs = torch.cat([batch_inputs, enhanced_inputs], dim=0)
        combined_feats = self.backbone(combined_inputs) 
        
        B = batch_inputs.size(0)
        raw_feats = [f[:B] for f in combined_feats]
        enh_feats = [f[B:] for f in combined_feats]
        
        x_raw, x_enh = self.neck(raw_feats, enh_feats)
        
        x_combined = [torch.cat([xr, xe], dim=0) for xr, xe in zip(x_raw, x_enh)]
        
        return x_combined