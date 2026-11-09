"""OAG with class-conditioned sonar prompts in RGB coordinates.

Opt-in detector: no behavior or parameters of the original OAG are changed.
"""
from copy import deepcopy

import torch
import torch.nn.functional as F

from mmdet.registry import MODELS
from .glip import clean_label_name
from .oag import OAG
from .sonarbind_prism import SonarBindPrism
from ..layers.transformer.oag_rgb_prompt_decoder import OAGRGBPromptDecoder
from ..utils.sonar_rgb_prompt import (
    SonarRGBPrompt, box_class_targets, prompt_token_bias, spatial_bce_dice)


@MODELS.register_module()
class OAGRGBPrompt(OAG):
    def __init__(self, prompt_level=2, prompt_temperature=0.07,
                 loss_rgb_prompt_weight=1.0, loss_sonar_class_weight=1.0,
                 **kwargs):
        self.prompt_level = prompt_level
        self.prompt_temperature = prompt_temperature
        self.loss_rgb_prompt_weight = loss_rgb_prompt_weight
        self.loss_sonar_class_weight = loss_sonar_class_weight
        super().__init__(**kwargs)
        if not self.use_text_branch or not self.use_language_guided_query:
            raise ValueError('OAGRGBPrompt requires text and language-guided queries')

    def _init_layers(self):
        cfg = deepcopy(self.decoder)
        super()._init_layers()
        del self.sonar_selector
        self.decoder = OAGRGBPromptDecoder(**cfg)
        for layer in self.decoder.layers:
            layer.rgb_prompt_attn.weight = self.decoder_sonar_attention_weight
        self.rgb_prompt_mapper = SonarRGBPrompt(self.embed_dims, self.prompt_temperature)

    def forward_encoder(self, feat, feat_mask, feat_pos, sonar_feat, sonar_mask,
                        sonar_pos, spatial_shapes, level_start_index, valid_ratios,
                        text_dict=None):
        memory, memory_text = self.encoder(
            query=feat, query_pos=feat_pos, key_padding_mask=feat_mask,
            spatial_shapes=spatial_shapes, level_start_index=level_start_index,
            valid_ratios=valid_ratios, memory_text=text_dict['embedded'],
            text_attention_mask=~text_dict['text_token_mask'],
            position_ids=text_dict.get('position_ids'),
            text_self_attention_masks=text_dict.get('masks'))
        return dict(memory=memory, memory_mask=feat_mask, spatial_shapes=spatial_shapes,
                    memory_text=memory_text, text_token_mask=text_dict['text_token_mask'],
                    memory_sonar=sonar_feat, memory_sonar_mask=sonar_mask)

    def _class_token_maps(self, samples, text_mask, dtype):
        maps = []
        for sample, valid_tokens in zip(samples, text_mask):
            # predict() supplies the exact map of the current prompt/chunk.
            if not self.training and 'token_positive_map' in sample:
                mapping = sample.token_positive_map
            else:
                caption = sample.get('text', sample.metainfo.get('classes'))
                if isinstance(caption, (list, tuple)):
                    caption = '. '.join(clean_label_name(c) for c in caption) + '.'
                if 'tokens_positive' in sample:
                    mapping, _, _, _ = self.get_tokens_positive_and_prompts(
                        caption, True, tokens_positive=sample.tokens_positive)
                else:
                    tokenized, _, spans, _ = self.get_tokens_and_prompts(caption, True)
                    mapping, _ = self.get_positive_map(tokenized, spans)
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError('RGB prompts require explicit, nonempty class/entity token spans')
            if sorted(mapping) != list(range(1, len(mapping) + 1)):
                raise ValueError('Entity token map must have consecutive one-based keys')
            matrix = torch.zeros(len(mapping), len(valid_tokens),
                                 device=text_mask.device, dtype=dtype)
            for label, indices in mapping.items():
                indices = [int(t) for t in indices if 0 <= int(t) < len(valid_tokens)
                           and bool(valid_tokens[int(t)])]
                if not indices:
                    raise ValueError('A prompt class has no valid tokens (possibly truncated)')
                matrix[label - 1, indices] = 1. / len(indices)
            maps.append(matrix)
        count = max(m.size(0) for m in maps)
        result = torch.stack([F.pad(m, (0, 0, 0, count - m.size(0))) for m in maps])
        return result, result.sum(-1) > 0

    def pre_decoder(self, memory, memory_mask, spatial_shapes, memory_text,
                    text_token_mask, memory_sonar=None, memory_sonar_mask=None,
                    batch_data_samples=None):
        bs, _, channels = memory.shape
        shapes = [tuple(int(v) for v in shape) for shape in spatial_shapes]
        if not 0 <= self.prompt_level < len(shapes):
            raise ValueError('prompt_level outside RGB feature pyramid')
        h, w = shapes[self.prompt_level]
        start = sum(a * b for a, b in shapes[:self.prompt_level])
        rgb = memory[:, start:start + h * w]
        if memory_mask is None:
            memory_mask = torch.zeros(memory.shape[:2], dtype=torch.bool, device=memory.device)
        mask = memory_mask[:, start:start + h * w]
        maps, class_valid = self._class_token_maps(batch_data_samples, text_token_mask, memory_text.dtype)
        classes = maps @ memory_text
        sh, sw = ((s + 31) // 32 for s in self.sonar_input_size)
        sonar_grid = memory_sonar.transpose(1, 2).reshape(bs, channels, sh, sw)
        prompt, sonar_logits, sonar_tokens, null = self.rgb_prompt_mapper(
            rgb, sonar_grid, classes, class_valid, mask, memory_sonar_mask)
        prompt_grid = prompt.transpose(1, 2).reshape(bs, -1, h, w)
        pyramid = torch.cat([
            F.interpolate(prompt_grid, size=shape, mode='bilinear', align_corners=False)
            .flatten(2).transpose(1, 2) for shape in shapes], 1)
        pyramid = pyramid.masked_fill(memory_mask[..., None], 0)
        positions = self.positional_encoding(mask.reshape(bs, h, w)).flatten(2).transpose(1, 2)

        output_memory, proposals = self.gen_encoder_output_proposals(memory, memory_mask, spatial_shapes)
        logits = self.bbox_head.cls_branches[self.decoder.num_layers](
            output_memory, memory_text, text_token_mask)
        coords = self.bbox_head.reg_branches[self.decoder.num_layers](output_memory) + proposals
        selection = logits + self.sonar_position_weight * prompt_token_bias(pyramid, maps, logits.size(-1))
        invalid = memory_mask | ~torch.isfinite(proposals).all(-1)
        scores = selection.max(-1).values.masked_fill(invalid, float('-inf'))
        if ((~invalid).sum(-1) < self.num_queries).any():
            raise ValueError('Fewer valid RGB proposals than num_queries')
        indices = scores.topk(self.num_queries, dim=1).indices
        top_logits = logits.gather(1, indices[..., None].expand(-1, -1, logits.size(-1)))
        top_coords = coords.gather(1, indices[..., None].expand(-1, -1, 4))
        query = self.query_embedding.weight[None].expand(bs, -1, -1)
        reference = top_coords.detach()
        if self.training:
            dn_query, dn_boxes, dn_mask, dn_meta = self.dn_query_generator(batch_data_samples)
            query = torch.cat([dn_query, query], 1)
            reference = torch.cat([dn_boxes, reference], 1)
        else:
            dn_mask, dn_meta = None, None
        decoder = dict(query=query, memory=memory, reference_points=reference.sigmoid(),
                       dn_mask=dn_mask, memory_text=memory_text,
                       text_attention_mask=~text_token_mask, rgb_prompt=prompt,
                       rgb_prompt_pos=positions, rgb_prompt_mask=mask,
                       prompt_classes=classes, prompt_class_valid=class_valid)
        head = dict(enc_outputs_class=top_logits, enc_outputs_coord=top_coords.sigmoid(),
                    dn_meta=dn_meta) if self.training else {}
        head.update(memory_text=memory_text, text_token_mask=text_token_mask,
                    memory_sonar=sonar_tokens, memory_sonar_mask=memory_sonar_mask,
                    memory=memory, memory_mask=memory_mask, spatial_shapes=spatial_shapes)
        if self.training:
            self._prompt_losses = self._position_losses(
                prompt, sonar_logits, class_valid, mask, memory_sonar_mask,
                batch_data_samples, (h, w), (sh, sw))
        return decoder, head

    def _compute_contrastive_losses(self, head_inputs_dict):
        # Optional legacy SonarBind CLIP-style losses. Their weights are zero
        # by default and are enabled by the dedicated contrastive config.
        if not self.enable_legacy_contrastive:
            return {}
        return SonarBindPrism._compute_contrastive_losses(self, head_inputs_dict)

    def _position_losses(self, prompt, sonar_logits, class_valid, rgb_mask,
                         sonar_mask, samples, rgb_size, sonar_size):
        rgb_targets, sonar_targets = [], []
        for sample, valid in zip(samples, class_valid):
            if 'sonar_gt_instances' not in sample:
                raise ValueError('Training requires independent sonar GT boxes and labels')
            count = int(valid.sum())
            sonar = box_class_targets(sample.sonar_gt_instances, sample.sonar_shape,
                                      sonar_size, count, prompt.device)
            # RGB boxes are augmented coordinates; grid spans the padded canvas.
            rgb = box_class_targets(sample.gt_instances, sample.batch_input_shape,
                                    rgb_size, count, prompt.device)
            present = torch.zeros(count, device=prompt.device)
            present[sample.sonar_gt_instances.labels.long()] = 1
            rgb = rgb * present[None]
            rgb_targets.append(F.pad(rgb, (0, prompt.size(-1) - count)))
            sonar_targets.append(F.pad(sonar, (0, prompt.size(-1) - count)))
        rgb_valid = (~rgb_mask)[..., None] & class_valid[:, None]
        sonar_valid = class_valid[:, None]
        if sonar_mask is not None:
            sonar_valid = sonar_valid & (~sonar_mask)[..., None]
        return dict(
            loss_rgb_prompt=self.loss_rgb_prompt_weight * spatial_bce_dice(
                prompt, torch.stack(rgb_targets), rgb_valid),
            loss_sonar_class=self.loss_sonar_class_weight * spatial_bce_dice(
                sonar_logits, torch.stack(sonar_targets), sonar_valid, logits=True))

    def loss(self, batch_inputs, batch_data_samples):
        self._prompt_losses = None
        losses = SonarBindPrism.loss(self, batch_inputs, batch_data_samples)
        if self._prompt_losses is None:
            raise RuntimeError('RGB prompt branch did not execute')
        losses.update(self._prompt_losses)
        self._prompt_losses = None
        return losses
