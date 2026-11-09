"""OAG with supervised sonar-to-optical affine plus residual coordinates."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from mmdet.registry import MODELS
from .oag import OAG


@MODELS.register_module()
class OAGAffine(OAG):
    def __init__(self, mapping_loss_weight=1., mapping_residual_limit=0.25,
                 query_position_weight=0.5, position_prior_sigma=0.1, **kwargs):
        self.mapping_loss_weight = mapping_loss_weight
        self.mapping_residual_limit = mapping_residual_limit
        self.query_position_weight = query_position_weight
        self.position_prior_sigma = position_prior_sigma
        if position_prior_sigma <= 0:
            raise ValueError('position_prior_sigma must be positive')
        super().__init__(**kwargs)

    def _init_layers(self):
        super()._init_layers()
        self.position_affine = nn.Linear(2, 2)
        self.position_residual = nn.Sequential(
            nn.Linear(self.embed_dims + 2, 128), nn.GELU(), nn.Linear(128, 2))
        self._reset_mapping()

    def _reset_mapping(self):
        nn.init.eye_(self.position_affine.weight)
        nn.init.zeros_(self.position_affine.bias)
        nn.init.zeros_(self.position_residual[-1].weight)
        nn.init.zeros_(self.position_residual[-1].bias)

    def init_weights(self):
        super().init_weights()
        self._reset_mapping()

    def forward_encoder(self, **kwargs):
        out = super().forward_encoder(**kwargs)
        features = out['memory_sonar']
        h, w = self._sonar_feature_size
        y, x = torch.meshgrid(
            (torch.arange(h, device=features.device, dtype=features.dtype)+.5)/h,
            (torch.arange(w, device=features.device, dtype=features.dtype)+.5)/w,
            indexing='ij')
        coords = torch.stack((x, y), -1).reshape(1, h*w, 2).expand(features.size(0), -1, -1)
        residual = self.mapping_residual_limit * self.position_residual(
            torch.cat((features, coords), -1)).tanh()
        self._mapped_coords = self.position_affine(coords) + residual
        return out

    def _image_coords(self, samples):
        # Mapping predicts canonical normalized RGB coordinates. Apply RGB
        # augmentation only afterwards; sonar is loaded after RGB augmentation.
        coords = self._mapped_coords
        flipped = []
        for c, sample in zip(coords, samples):
            c = c.clone()
            if sample.get('flip', False):
                direction = sample.get('flip_direction', 'horizontal')
                if direction in ('horizontal', 'diagonal'):
                    c = torch.stack((1-c[:, 0], c[:, 1]), -1)
                if direction in ('vertical', 'diagonal'):
                    c = torch.stack((c[:, 0], 1-c[:, 1]), -1)
            flipped.append(c)
        return torch.stack(flipped)

    def _encode_positions(self, coords):
        # Same y/x sinusoidal ordering and temperature as the OAG sine encoder.
        n = self.embed_dims // 2
        dim = torch.arange(n, device=coords.device, dtype=coords.dtype)
        scale = self.positional_encoding.temperature ** (2 * (dim // 2) / n)
        parts = []
        for axis in (1, 0):
            phase = coords[..., axis, None] * (2 * math.pi) / scale
            parts.append(torch.stack((phase[..., 0::2].sin(),
                                      phase[..., 1::2].cos()), -1).flatten(-2))
        return torch.cat(parts, -1)

    def _position_scores(self, centers, mapped, weights):
        # Fixed-width Gaussian mixture, not learned uncertainty. Chunk proposals
        # to avoid allocating a full proposal x sonar-token tensor at once.
        result = []
        for chunk in centers.split(512, dim=1):
            dist = (chunk[:, :, None] - mapped[:, None]).square().sum(-1)
            response = torch.exp(-dist / (2 * self.position_prior_sigma**2))
            result.append((response * weights[:, None]).sum(-1) /
                          weights.sum(-1, keepdim=True).clamp_min(1e-6))
        return torch.cat(result, 1)

    def loss(self, batch_inputs, batch_data_samples):
        losses = super().loss(batch_inputs, batch_data_samples)
        mapped = self._image_coords(batch_data_samples)
        terms = []
        h, w = self._sonar_feature_size
        for i, sample in enumerate(batch_data_samples):
            sonar, rgb = sample.sonar_gt_instances, sample.gt_instances
            for label in sonar.labels.unique():
                si, ri = sonar.labels == label, rgb.labels == label
                # Same-class multiple objects need instance correspondence.
                if int(si.sum()) != 1 or int(ri.sum()) != 1:
                    continue
                sb = sonar.bboxes[si][0].to(mapped)
                rb = rgb.bboxes[ri][0]
                if hasattr(rb, 'tensor'):
                    rb = rb.tensor.squeeze(0)
                rb = rb.to(mapped)
                sh, sw = sample.sonar_shape[:2]
                ih, iw = sample.img_shape[:2]
                center = (sb[:2] + sb[2:]) / sb.new_tensor([2*sw, 2*sh])
                prediction = F.grid_sample(
                    mapped[i].T.reshape(1, 2, h, w),
                    (center * 2 - 1).reshape(1, 1, 1, 2),
                    align_corners=False, padding_mode='border').reshape(2)
                target = (rb[:2] + rb[2:]) / rb.new_tensor([2*iw, 2*ih])
                terms.append(F.smooth_l1_loss(prediction, target, beta=0.05))
        losses['loss_sonar_position_map'] = self.mapping_loss_weight * (
            torch.stack(terms).mean() if terms else mapped.sum() * 0)
        return losses

    def pre_decoder(
        self,
        memory,
        memory_mask,
        spatial_shapes,
        memory_text,
        text_token_mask,
        memory_sonar = None,
        memory_sonar_mask = None,
        batch_data_samples=None,
        sonar_pos=None, sonar_selection_weights=None,
    ):
        bs, _, c = memory.shape
        mapped = self._image_coords(batch_data_samples)
        # Decoder references are in padded RGB coordinates.
        factors = mapped.new_tensor([
            [s.img_shape[1] / s.batch_input_shape[1],
             s.img_shape[0] / s.batch_input_shape[0]]
            for s in batch_data_samples])
        sonar_pos = self._encode_positions(mapped * factors[:, None])


        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)

        if self.use_language_guided_query and memory_text is not None:
            # Original logic: language-guided query selection
            enc_outputs_class = self.bbox_head.cls_branches[
                self.decoder.num_layers](output_memory, memory_text,
                                         text_token_mask)
            cls_out_features = self.bbox_head.cls_branches[
                self.decoder.num_layers].max_text_len
            enc_outputs_coord_unact = self.bbox_head.reg_branches[
                self.decoder.num_layers](output_memory) + output_proposals

            # NOTE The DINO selects top-k proposals according to scores of
            # multi-class classification, while DeformDETR, where the input
            # is `enc_outputs_class[..., 0]` selects according to scores of
            # binary classification.
            centers = enc_outputs_coord_unact[..., :2].sigmoid()
            prior = self._position_scores(centers, mapped, sonar_selection_weights)
            scores = enc_outputs_class.max(-1)[0] + self.query_position_weight * prior
            topk_indices = torch.topk(scores, k=self.num_queries, dim=1)[1]

            topk_score = torch.gather(
                enc_outputs_class, 1,
                topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        else:
            # Ablation logic: fixed query selection based on objectness
            # Use L2 norm of encoder output features as objectness score
            objectness_scores = output_memory.norm(dim=-1)  # [bs, num_proposals]

            enc_outputs_coord_unact = self.bbox_head.reg_branches[
                self.decoder.num_layers](output_memory) + output_proposals

            # Select top-k based on objectness
            topk_indices = torch.topk(
                objectness_scores, k=self.num_queries, dim=1)[1]

            # Create dummy topk_score (not used in inference without text)
            topk_score = None
            cls_out_features = 256  # dummy value

        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        topk_coords = topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()

        query = self.query_embedding.weight[:, None, :]
        query = query.repeat(1, bs, 1).transpose(0, 1)
        if self.training:
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
                self.dn_query_generator(batch_data_samples)
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
            dn_mask=dn_mask,
            memory_text=memory_text,
            text_attention_mask=~text_token_mask if text_token_mask is not None else None,
        )
        # NOTE DINO calculates encoder losses on scores and coordinates
        # of selected top-k encoder queries, while DeformDETR is of all
        # encoder queries.
        head_inputs_dict = dict(
            enc_outputs_class=topk_score,
            enc_outputs_coord=topk_coords,
            dn_meta=dn_meta) if self.training else dict()
        # append text_feats to head_inputs_dict
        head_inputs_dict['memory_text'] = memory_text
        head_inputs_dict['text_token_mask'] = text_token_mask
        # append sonar_feats to head_inputs_dict (sonar不进decoder，与memory_text对称保存)
        head_inputs_dict['memory_sonar'] = memory_sonar
        head_inputs_dict['memory_sonar_mask'] = memory_sonar_mask
        head_inputs_dict['memory'] = memory
        head_inputs_dict['memory_mask'] = memory_mask
        head_inputs_dict['spatial_shapes'] = spatial_shapes
        decoder_inputs_dict.update(
            sonar_pos=sonar_pos, sonar_selection_weights=sonar_selection_weights,
            memory_sonar=memory_sonar, memory_sonar_mask=memory_sonar_mask)
        return decoder_inputs_dict, head_inputs_dict
