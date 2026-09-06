# Copyright (c) OpenMMLab. All rights reserved.
"""
GroundingDINOHeadNoText: A variant of GroundingDINOHead for ablation studies
without text branch. Uses traditional Linear classification layers instead of
ContrastiveEmbed for text-vision matching.

This file is designed for ablation experiments and does not modify existing source files.
"""

import copy
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import Linear
from mmengine.model import constant_init
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
from mmdet.utils import InstanceList
from ..layers import inverse_sigmoid
from .dino_head import DINOHead


@MODELS.register_module()
class GroundingDINOHeadNoText(DINOHead):
    """Head of Grounding DINO without text branch for ablation studies.

    This head uses traditional Linear classification layers instead of
    ContrastiveEmbed, making it suitable for experiments without language guidance.

    Args:
        num_classes (int): Number of object categories.
        All other args are inherited from DINOHead.
    """

    def __init__(self, num_classes: int = 80, **kwargs):
        # Pass num_classes to parent class
        super().__init__(num_classes=num_classes, **kwargs)
        self.num_classes = num_classes

    def _init_layers(self) -> None:
        """Initialize classification branch and regression branch of head."""
        # Use traditional Linear layer for classification instead of ContrastiveEmbed
        fc_cls = Linear(self.embed_dims, self.num_classes)

        # Regression branch (same as GroundingDINOHead)
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, 4))
        reg_branch = nn.Sequential(*reg_branch)

        if self.share_pred_layer:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(self.num_pred_layer)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(self.num_pred_layer)])
        else:
            self.cls_branches = nn.ModuleList(
                [copy.deepcopy(fc_cls) for _ in range(self.num_pred_layer)])
            self.reg_branches = nn.ModuleList([
                copy.deepcopy(reg_branch) for _ in range(self.num_pred_layer)
            ])

    def init_weights(self) -> None:
        """Initialize weights of the head."""
        # Initialize classification branches
        for m in self.cls_branches:
            if hasattr(m, 'weight'):
                nn.init.normal_(m.weight, mean=0, std=0.01)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

        # Initialize regression branches
        for m in self.reg_branches:
            constant_init(m[-1], 0, bias=0)
        nn.init.constant_(self.reg_branches[0][-1].bias.data[2:], -2.0)
        if self.as_two_stage:
            for m in self.reg_branches:
                nn.init.constant_(m[-1].bias.data[2:], 0.0)

    def forward(
        self,
        hidden_states: Tensor,
        references: List[Tensor],
        memory_text: Tensor = None,
        text_token_mask: Tensor = None,
    ) -> Tuple[Tensor]:
        """Forward function.

        Note: memory_text and text_token_mask are kept in signature for
        compatibility but are not used in this no-text variant.

        Args:
            hidden_states (Tensor): Hidden states output from each decoder
                layer, has shape (num_decoder_layers, bs, num_queries, dim).
            references (List[Tensor]): List of the reference from the decoder.
            memory_text (Tensor, optional): Not used. Kept for compatibility.
            text_token_mask (Tensor, optional): Not used. Kept for compatibility.

        Returns:
            tuple[Tensor]: results of head containing the following tensor.

            - all_layers_outputs_classes (Tensor): Outputs from the
              classification head, has shape (num_decoder_layers, bs,
              num_queries, num_classes).
            - all_layers_outputs_coords (Tensor): Sigmoid outputs from the
              regression head with normalized coordinate format (cx, cy, w,
              h), has shape (num_decoder_layers, bs, num_queries, 4).
        """
        all_layers_outputs_classes = []
        all_layers_outputs_coords = []

        for layer_id in range(hidden_states.shape[0]):
            reference = inverse_sigmoid(references[layer_id])
            hidden_state = hidden_states[layer_id]

            # Use Linear layer directly without text features
            outputs_class = self.cls_branches[layer_id](hidden_state)

            tmp_reg_preds = self.reg_branches[layer_id](hidden_state)
            if reference.shape[-1] == 4:
                tmp_reg_preds += reference
            else:
                assert reference.shape[-1] == 2
                tmp_reg_preds[..., :2] += reference
            outputs_coord = tmp_reg_preds.sigmoid()

            all_layers_outputs_classes.append(outputs_class)
            all_layers_outputs_coords.append(outputs_coord)

        all_layers_outputs_classes = torch.stack(all_layers_outputs_classes)
        all_layers_outputs_coords = torch.stack(all_layers_outputs_coords)

        return all_layers_outputs_classes, all_layers_outputs_coords

    def loss(self, hidden_states: Tensor, references: List[Tensor],
             memory_text: Tensor, text_token_mask: Tensor,
             enc_outputs_class: Tensor, enc_outputs_coord: Tensor,
             batch_data_samples: SampleList, dn_meta: Dict[str, int]) -> dict:
        """Perform forward propagation and loss calculation.

        Args:
            hidden_states (Tensor): Hidden states from decoder layers.
            references (list[Tensor]): References from the decoder.
            memory_text (Tensor): Not used, kept for compatibility.
            text_token_mask (Tensor): Not used, kept for compatibility.
            enc_outputs_class (Tensor): Encoder output classifications.
            enc_outputs_coord (Tensor): Encoder output coordinates.
            batch_data_samples (list[:obj:`DetDataSample`]): Batch data samples.
            dn_meta (Dict[str, int]): Denoising meta information.

        Returns:
            dict: A dictionary of loss components.
        """
        batch_gt_instances = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)

        # Forward pass (memory_text and text_token_mask ignored)
        outs = self(hidden_states, references, memory_text, text_token_mask)

        loss_inputs = outs + (enc_outputs_class, enc_outputs_coord,
                              batch_gt_instances, batch_img_metas, dn_meta)
        losses = self.loss_by_feat(*loss_inputs)
        return losses

    def _get_targets_single(self, cls_score: Tensor, bbox_pred: Tensor,
                            gt_instances: InstanceData,
                            img_meta: dict) -> tuple:
        """Compute regression and classification targets for one image.

        This uses standard class labels instead of positive maps.

        Args:
            cls_score (Tensor): Box score logits, shape [num_queries, num_classes].
            bbox_pred (Tensor): Bbox predictions, shape [num_queries, 4].
            gt_instances (:obj:`InstanceData`): Ground truth instances.
            img_meta (dict): Meta information for one image.

        Returns:
            tuple[Tensor]: a tuple containing targets for one image.
        """
        img_h, img_w = img_meta['img_shape']
        factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)
        num_bboxes = bbox_pred.size(0)

        # Convert bbox_pred from xywh, normalized to xyxy, unnormalized
        bbox_pred = bbox_cxcywh_to_xyxy(bbox_pred)
        bbox_pred = bbox_pred * factor

        pred_instances = InstanceData(scores=cls_score, bboxes=bbox_pred)

        # Assigner and sampler
        assign_result = self.assigner.assign(
            pred_instances=pred_instances,
            gt_instances=gt_instances,
            img_meta=img_meta)
        gt_bboxes = gt_instances.bboxes

        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
        pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds.long(), :]

        # Use standard class labels (num_classes) instead of text token labels
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    self.num_classes,  # background class
                                    dtype=torch.long)
        labels[pos_inds] = gt_instances.labels[pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # Bbox targets
        bbox_targets = torch.zeros_like(bbox_pred, dtype=gt_bboxes.dtype)
        bbox_weights = torch.zeros_like(bbox_pred, dtype=gt_bboxes.dtype)
        bbox_weights[pos_inds] = 1.0

        # Normalize bbox targets
        pos_gt_bboxes_normalized = pos_gt_bboxes / factor
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        bbox_targets[pos_inds] = pos_gt_bboxes_targets

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds)

    def predict(self,
                hidden_states: Tensor,
                references: List[Tensor],
                memory_text: Tensor,
                text_token_mask: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> InstanceList:
        """Perform forward propagation for prediction.

        Args:
            hidden_states (Tensor): Hidden states from decoder.
            references (List[Tensor]): References from decoder.
            memory_text (Tensor): Not used, kept for compatibility.
            text_token_mask (Tensor): Not used, kept for compatibility.
            batch_data_samples (SampleList): Batch data samples.
            rescale (bool, optional): If `True`, return boxes in original
                image space. Defaults to `True`.

        Returns:
            InstanceList: Detection results of each image.
        """
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        outs = self(hidden_states, references, memory_text, text_token_mask)

        predictions = self.predict_by_feat(
            *outs,
            batch_img_metas=batch_img_metas,
            rescale=rescale)
        return predictions

    def predict_by_feat(self,
                        all_layers_cls_scores: Tensor,
                        all_layers_bbox_preds: Tensor,
                        batch_img_metas: List[Dict],
                        rescale: bool = False) -> InstanceList:
        """Transform output features into bbox results.

        Args:
            all_layers_cls_scores (Tensor): Classification scores.
            all_layers_bbox_preds (Tensor): Bbox predictions.
            batch_img_metas (List[Dict]): Batch image metas.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            list[:obj:`InstanceData`]: Detection results.
        """
        cls_scores = all_layers_cls_scores[-1]
        bbox_preds = all_layers_bbox_preds[-1]

        result_list = []
        for img_id in range(len(batch_img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            img_meta = batch_img_metas[img_id]

            results = self._predict_by_feat_single(
                cls_score, bbox_pred, img_meta, rescale)
            result_list.append(results)

        return result_list

    def _predict_by_feat_single(self,
                                cls_score: Tensor,
                                bbox_pred: Tensor,
                                img_meta: dict,
                                rescale: bool = True) -> InstanceData:
        """Transform single image output into detection results.

        Args:
            cls_score (Tensor): Classification scores, shape (num_queries, num_classes).
            bbox_pred (Tensor): Bbox predictions, shape (num_queries, 4).
            img_meta (dict): Image meta information.
            rescale (bool): Whether to rescale bboxes.

        Returns:
            :obj:`InstanceData`: Detection results.
        """
        assert len(cls_score) == len(bbox_pred)

        max_per_img = self.test_cfg.get('max_per_img', 100)

        # Apply sigmoid and get scores
        scores = cls_score.sigmoid()
        scores, labels = scores.max(dim=-1)

        # Filter by score threshold
        if self.test_cfg.get('score_thr', 0) > 0:
            score_thr = self.test_cfg['score_thr']
            valid_mask = scores > score_thr
            scores = scores[valid_mask]
            labels = labels[valid_mask]
            bbox_pred = bbox_pred[valid_mask]

        # Top-k selection
        if len(scores) > max_per_img:
            scores, indices = scores.topk(max_per_img)
            labels = labels[indices]
            bbox_pred = bbox_pred[indices]

        # Rescale bboxes
        img_shape = img_meta['img_shape']
        bbox_pred = bbox_cxcywh_to_xyxy(bbox_pred)
        bbox_pred[:, 0::2] = bbox_pred[:, 0::2] * img_shape[1]
        bbox_pred[:, 1::2] = bbox_pred[:, 1::2] * img_shape[0]

        if rescale:
            scale_factor = img_meta['scale_factor']
            bbox_pred /= bbox_pred.new_tensor(scale_factor).repeat(2)

        results = InstanceData()
        results.bboxes = bbox_pred
        results.scores = scores
        results.labels = labels

        return results
