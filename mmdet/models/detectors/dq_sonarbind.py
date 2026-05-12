# Copyright (c) OpenMMLab. All rights reserved.
"""DQSonarBind: SonarBind with DQ-DETR innovations.

Three DQ-DETR innovations are added on top of SonarBind:
  1. CCM  (Categorical Counting Module)      -- predicts scene density class
  2. CGFE (Context-Guided Feature Enhancement) -- enhances encoder memory
  3. Dynamic Query Selection                 -- adapts num_queries to density

All backbone / sonar / text / contrastive-loss specifics of SonarBind are
preserved without modification.  Only `_init_layers` and `pre_decoder` are
overridden; everything else is inherited as-is.
"""

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList
from mmdet.utils import OptConfigType

from mmdet.models.utils import CategoricalCounting, CGFE, MultiScaleFeature
from .sonarbind import SonarBind


@MODELS.register_module()
class DQSonarBind(SonarBind):
    """SonarBind augmented with DQ-DETR dynamic-query mechanisms.

    New constructor args (all optional with sensible defaults):
        ccm_cfg (dict): kwargs forwarded to CategoricalCounting.
        cgfe_cfg (dict): kwargs forwarded to CGFE.
        dynamic_query_list (list[int]): one num_queries value per CCM class.
            Length must equal ccm_cfg['cls_num'].  Defaults to
            [300, 500, 700, 900].
    """

    def __init__(
        self,
        *args,
        ccm_cfg: OptConfigType = None,
        cgfe_cfg: OptConfigType = None,
        dynamic_query_list=None,
        **kwargs,
    ) -> None:
        # Store DQ params BEFORE calling super().__init__ so that
        # _init_layers (called inside super) can access them.
        self._ccm_cfg = ccm_cfg or {}
        self._cgfe_cfg = cgfe_cfg or {}
        self._dynamic_query_list = (
            dynamic_query_list if dynamic_query_list is not None
            else [300, 500, 700, 900]
        )
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Layer initialisation
    # ------------------------------------------------------------------

    def _init_layers(self) -> None:
        """Call parent init, then attach the three DQ modules."""
        super()._init_layers()

        # CCM: density classifier operating on the finest encoder feature map
        self.ccm = CategoricalCounting(**self._ccm_cfg)

        # MultiScaleFeature: generates one CCM-derived feature per FPN level
        is_5_scale = (self.num_feature_levels == 5)
        self.multiscale = MultiScaleFeature(is_5_scale=is_5_scale)

        # CGFE: context-guided spatial+channel gating of encoder memory
        self.cgfe = CGFE(**self._cgfe_cfg)

        # Dynamic query budget table (one entry per CCM density class)
        self.dynamic_query_list = self._dynamic_query_list

        # Tracks the query count used in the latest forward pass (for debug)
        self.current_num_queries = self.num_queries

    # ------------------------------------------------------------------
    # pre_decoder override -- the single point of change
    # ------------------------------------------------------------------

    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        memory_text: Tensor,
        text_token_mask: Tensor,
        memory_sonar: Optional[Tensor] = None,
        memory_sonar_mask: Optional[Tensor] = None,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict, Dict]:
        """pre_decoder with CCM / CGFE / dynamic-query injected.

        Differences from SonarBind.pre_decoder:
        * memory is enhanced by CGFE before proposal generation.
        * num_select is chosen dynamically from dynamic_query_list based on
          the scene-density class predicted by CCM.
        * dn_query_generator.num_matching_queries is temporarily patched to
          match num_select so that DN-query tensors have the right shape.
        * A tiny (1e-10) CCM gradient hook is added to output_memory so that
          CCM parameters always participate in the backward graph (required
          when static_graph=True in DDP).

        All other logic (text-aware classification, contrastive-loss fields,
        sonar fields in head_inputs_dict, etc.) is identical to SonarBind.
        """

        # ----------------------------------------------------------------
        # Step 1: CCM -- predict scene density from finest feature level
        # ----------------------------------------------------------------
        counting_output, ccm_feature = self.ccm(memory, spatial_shapes)
        # counting_output : (bs, cls_num)
        # ccm_feature     : (bs, 256, H0, W0)  [finest scale spatial map]

        # ----------------------------------------------------------------
        # Step 2: CGFE -- enhance encoder memory with CCM context
        # ----------------------------------------------------------------
        multi_ccm_feature = self.multiscale(ccm_feature)
        memory = self.cgfe(multi_ccm_feature, memory, spatial_shapes)
        # memory : (bs, sum_HW, 256)  -- enhanced, same shape as before

        # ----------------------------------------------------------------
        # Step 3: Dynamic query count
        # ----------------------------------------------------------------
        _, predicted_density_idx = torch.max(counting_output, dim=1)
        # Use the most-crowded image's class to size the whole batch
        batch_max_idx = int(predicted_density_idx.max().item())
        num_select = self.dynamic_query_list[batch_max_idx]
        self.current_num_queries = num_select

        # ----------------------------------------------------------------
        # Step 4: Proposal generation (identical to SonarBind)
        # ----------------------------------------------------------------
        bs, _, c = memory.shape

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)

        # ----------------------------------------------------------------
        # Step 5: DDP gradient hook for CCM
        # Ensures CCM parameters always receive gradients when static_graph=True.
        # The 1e-10 coefficient makes the contribution numerically negligible.
        # ----------------------------------------------------------------
        if self.training:
            output_memory = output_memory + counting_output.sum() * 1e-10

        # ----------------------------------------------------------------
        # Step 6: Text-aware classification & regression (SonarBind-specific)
        # ----------------------------------------------------------------
        enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers](
                output_memory, memory_text, text_token_mask)
        cls_out_features = self.bbox_head.cls_branches[
            self.decoder.num_layers].max_text_len
        enc_outputs_coord_unact = (
            self.bbox_head.reg_branches[self.decoder.num_layers](output_memory)
            + output_proposals)

        # ----------------------------------------------------------------
        # Step 7: Top-k selection with dynamic num_select
        # ----------------------------------------------------------------
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

        # ----------------------------------------------------------------
        # Step 8: Query embedding + DN queries
        # ----------------------------------------------------------------
        query = self.query_embedding.weight[:num_select, None, :]
        query = query.repeat(1, bs, 1).transpose(0, 1)

        if self.training:
            # Temporarily patch num_matching_queries so DN-query tensors
            # match the dynamic num_select (restored immediately after).
            orig_num = self.dn_query_generator.num_matching_queries
            self.dn_query_generator.num_matching_queries = num_select

            dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
                self.dn_query_generator(batch_data_samples)

            self.dn_query_generator.num_matching_queries = orig_num

            query = torch.cat([dn_label_query, query], dim=1)
            reference_points = torch.cat(
                [dn_bbox_query, topk_coords_unact], dim=1)
        else:
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None

        reference_points = reference_points.sigmoid()

        # ----------------------------------------------------------------
        # Step 9: Assemble dicts (identical structure to SonarBind)
        # ----------------------------------------------------------------
        decoder_inputs_dict = dict(
            query=query,
            memory=memory,
            reference_points=reference_points,
            dn_mask=dn_mask,
            memory_text=memory_text,
            text_attention_mask=~text_token_mask,
        )

        head_inputs_dict = dict(
            enc_outputs_class=topk_score,
            enc_outputs_coord=topk_coords,
            dn_meta=dn_meta) if self.training else dict()

        # Preserve all SonarBind-specific fields in head_inputs_dict
        head_inputs_dict['memory_text'] = memory_text
        head_inputs_dict['text_token_mask'] = text_token_mask
        head_inputs_dict['memory_sonar'] = memory_sonar
        head_inputs_dict['memory_sonar_mask'] = memory_sonar_mask
        head_inputs_dict['memory'] = memory
        head_inputs_dict['memory_mask'] = memory_mask
        head_inputs_dict['spatial_shapes'] = spatial_shapes

        return decoder_inputs_dict, head_inputs_dict
