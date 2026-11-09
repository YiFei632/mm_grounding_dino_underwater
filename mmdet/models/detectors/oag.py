import torch
import torch.nn.functional as F
from torch import Tensor
from mmdet.registry import MODELS
from .sonarbind_prism import SonarBindPrism
from ..layers.transformer.grounding_dino_layers import GroundingDinoTransformerEncoder
from ..utils.sonar_feature_selector import SonarFeatureSelector, sonar_box_mask


@MODELS.register_module()
class OAG(SonarBindPrism):
    """Optical-language Grounding with text-selected sonar position prompts."""
    def __init__(self, sonar_selector=None, sonar_position_weight=0.5,
                 decoder_sonar_cross_attn=False,
                 decoder_sonar_attention_weight=0.5,
                 decoder_sonar_mode='feature',
                 loss_sonar_image_weight=0.0,
                 loss_sonar_text_weight=0.0,
                 enable_legacy_contrastive=False,
                 loss_sonar_select_weight=1.0, loss_sonar_select_bce_weight=1.0,
                 loss_sonar_select_dice_weight=1.0, **kwargs):
        self.decoder_sonar_cross_attn = decoder_sonar_cross_attn
        if decoder_sonar_mode not in ('feature', 'position_only'):
            raise ValueError('Unknown decoder_sonar_mode')
        self.decoder_sonar_mode = decoder_sonar_mode
        self.decoder_sonar_attention_weight = decoder_sonar_attention_weight
        self.enable_legacy_contrastive = enable_legacy_contrastive
        self.sonar_selector_cfg = sonar_selector or dict(type='SonarFeatureSelector', in_channels=256)
        self.encoder_cfg = kwargs.get('encoder', {}).copy()
        self.encoder_cfg.pop('type', None)
        for key in ('fusion_image_sonar_cfg', 'sonar_layer_cfg', 'use_text_branch'):
            self.encoder_cfg.pop(key, None)
        self.sonar_position_weight = sonar_position_weight
        self.loss_sonar_select_weight = loss_sonar_select_weight
        self.loss_sonar_select_bce_weight = loss_sonar_select_bce_weight
        self.loss_sonar_select_dice_weight = loss_sonar_select_dice_weight
        # Keep the legacy contrastive terms opt-in. Existing OAG configs keep
        # their previous behavior; derived models can expose these weights.
        kwargs['loss_sonar_image_weight'] = (loss_sonar_image_weight
                                             if enable_legacy_contrastive else 0.0)
        kwargs['loss_sonar_text_weight'] = (loss_sonar_text_weight
                                            if enable_legacy_contrastive else 0.0)
        super().__init__(**kwargs)

    def _init_layers(self):
        from copy import deepcopy
        decoder_cfg = deepcopy(self.decoder) if self.decoder_sonar_cross_attn else None
        super()._init_layers()
        if self.decoder_sonar_cross_attn:
            from ..layers.transformer.oag_decoder import OAGSonarDecoder
            self.decoder = OAGSonarDecoder(**decoder_cfg)
            for layer in self.decoder.layers:
                layer.sonar_attention_weight = self.decoder_sonar_attention_weight
                layer.sonar_mode = self.decoder_sonar_mode
        # Replace the SonarBind encoder with the original Grounding-DINO
        # optical/text encoder.  Sonar is kept as an auxiliary stream.
        self.encoder = GroundingDinoTransformerEncoder(**self.encoder_cfg)
        self.sonar_selector = MODELS.build(self.sonar_selector_cfg)

    def forward_encoder(self, feat, feat_mask, feat_pos, sonar_feat, sonar_mask,
                        sonar_pos, spatial_shapes, level_start_index, valid_ratios,
                        text_dict=None):
        # Both training and inference call this method (training bypasses
        # forward_transformer). ResNet stage 4 has stride 32.
        h, w = ((size + 31) // 32 for size in self.sonar_input_size)
        if h * w != sonar_feat.shape[1]:
            raise ValueError('Sonar token shape does not match ResNet stage-4 grid')
        sonar_grid = sonar_feat.transpose(1, 2).reshape(
            sonar_feat.shape[0], sonar_feat.shape[2], h, w)
        sonar_feat, logits, weights = self.sonar_selector(
            sonar_grid, text_dict['embedded'], text_dict['text_token_mask'])
        self._last_sonar_selection_logits = logits
        self._sonar_feature_size = (h, w)
        self._last_sonar_tokens = sonar_feat
        # Grounding-DINO visual/text encoder; sonar remains an auxiliary branch.
        encoded = self.encoder(
            query=feat, query_pos=feat_pos, key_padding_mask=feat_mask,
            spatial_shapes=spatial_shapes, level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            memory_text=text_dict['embedded'] if text_dict else None,
            text_attention_mask=~text_dict['text_token_mask'] if text_dict else None,
            position_ids=text_dict.get('position_ids') if text_dict else None,
            text_self_attention_masks=text_dict.get('masks') if text_dict else None)
        if isinstance(encoded, tuple):
            memory, memory_text = encoded
        else:
            memory, memory_text = encoded, text_dict['embedded'] if text_dict else None
        self._last_memory_text = memory_text
        return dict(memory=memory, memory_mask=feat_mask, spatial_shapes=spatial_shapes,
                    memory_text=memory_text,
                    text_token_mask=text_dict['text_token_mask'] if text_dict else None,
                    memory_sonar=sonar_feat, memory_sonar_mask=sonar_mask,
                    sonar_pos=sonar_pos, sonar_selection_weights=weights)

    def pre_decoder(self, memory, memory_mask, spatial_shapes, memory_text,
                    text_token_mask, memory_sonar=None, memory_sonar_mask=None,
                    sonar_pos=None, batch_data_samples=None,
                    sonar_selection_weights=None):
        out = super().pre_decoder(memory, memory_mask, spatial_shapes, memory_text,
                                  text_token_mask, memory_sonar, memory_sonar_mask,
                                  batch_data_samples)
        dec, head = out
        if self.decoder_sonar_cross_attn:
            dec.update(memory_sonar=memory_sonar, sonar_pos=sonar_pos,
                       memory_sonar_mask=memory_sonar_mask,
                       sonar_selection_weights=sonar_selection_weights)
        return dec, head

    def _compute_contrastive_losses(self, head_inputs_dict):
        losses = super()._compute_contrastive_losses(head_inputs_dict)
        losses.pop('loss_sonar_image', None)
        return losses

    def loss(self, batch_inputs, batch_data_samples):
        """Add independent-sonar-box supervision to the normal OAG losses."""
        self._last_sonar_selection_logits = None
        losses = super().loss(batch_inputs, batch_data_samples)
        logits = self._last_sonar_selection_logits
        if logits is None:
            raise RuntimeError('OAG selector was not executed in the training forward')
        losses['loss_sonar_select'] = self._sonar_selection_loss(
            logits, batch_data_samples, self.sonar_input_size,
            self._sonar_feature_size)
        losses['loss_sonar_text'] = self._gt_sonar_text_loss(
            self._last_sonar_tokens, self._last_memory_text,
            batch_data_samples, self._sonar_feature_size)
        return losses

    def _gt_sonar_text_loss(self, sonar_tokens, text_tokens, samples, feat_size):
        """GT-supervised contrast: sonar GT regions against class text tokens."""
        if text_tokens is None:
            return sonar_tokens.sum() * 0.0
        h, w = feat_size
        all_regions, all_labels = [], []
        classes = getattr(samples[0], 'metainfo', {}).get('classes', None)
        if classes is None:
            classes = ('connected_polyhedron', 'fake_person', 'frustum',
                       'iron_ball', 'octahedron', 'uuv')
        if self.language_model is None or self.text_feat_map is None:
            return sonar_tokens.sum() * 0.0
        prompts = [str(c).replace('_', ' ') + '.' for c in classes]
        class_dict = self.language_model(prompts)
        class_emb = self.text_feat_map(class_dict['embedded'])
        cmask = class_dict.get('text_token_mask', None)
        if cmask is not None:
            class_emb = (class_emb * cmask.unsqueeze(-1).to(class_emb.dtype)).sum(1) / cmask.sum(1, keepdim=True).clamp_min(1)
        else:
            class_emb = class_emb.mean(1)
        for i, sample in enumerate(samples):
            inst = getattr(sample, 'sonar_gt_instances', None)
            boxes = getattr(inst, 'bboxes', None) if inst is not None else None
            labels = getattr(inst, 'labels', None) if inst is not None else None
            if boxes is None or labels is None or len(boxes) == 0:
                continue
            feat = sonar_tokens[i].reshape(h, w, -1)
            for box, label in zip(boxes, labels):
                x1, y1, x2, y2 = box
                iw, ih = sample.sonar_shape[1], sample.sonar_shape[0]
                xa, xb = int(x1 / iw * w), max(int(x2 / iw * w), int(x1 / iw * w) + 1)
                ya, yb = int(y1 / ih * h), max(int(y2 / ih * h), int(y1 / ih * h) + 1)
                xa, xb = max(0, xa), min(w, xb); ya, yb = max(0, ya), min(h, yb)
                if xa >= xb or ya >= yb: continue
                all_regions.append(feat[ya:yb, xa:xb].mean((0, 1)))
                all_labels.append(label.to(sonar_tokens.device))
        if not all_regions:
            return sonar_tokens.sum() * 0.0
        sonar = F.normalize(torch.stack(all_regions), dim=-1)
        text = F.normalize(class_emb.to(sonar.device), dim=-1)
        logits = sonar @ text.T / self.contrastive_temperature
        target = torch.stack(all_labels).long().to(logits.device)
        return F.cross_entropy(logits, target) * 0.1

    def _sonar_selection_loss(self, logits, batch_data_samples, image_size, feat_size):
        targets = []
        for sample in batch_data_samples:
            inst = getattr(sample, 'sonar_gt_instances', None)
            boxes = getattr(inst, 'bboxes', None) if inst is not None else None
            if boxes is None:
                raise ValueError('OAG training requires sonar_gt_instances.bboxes')
            else:
                targets.append(sonar_box_mask(boxes.to(logits.device), sample.sonar_shape, feat_size))
        target = torch.stack(targets).flatten(1)
        bce = F.binary_cross_entropy_with_logits(logits, target)
        prob = logits.sigmoid()
        dice = 1 - (2 * (prob * target).sum(1) + 1) / (prob.sum(1) + target.sum(1) + 1)
        return self.loss_sonar_select_weight * (self.loss_sonar_select_bce_weight * bce + self.loss_sonar_select_dice_weight * dice.mean())
