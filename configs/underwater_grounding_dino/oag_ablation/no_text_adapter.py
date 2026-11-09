"""Config-local, closed-set modality ablation. No changes to mmdet sources.

Learned category vectors replace language embeddings. Existing GroundingDINOHead
is used as a learned prototype classifier, with one channel per category.
The variables memory_text/text_token_mask are compatibility names, not text.
"""
import torch
from torch import nn
import torch.nn.functional as F

from mmdet.registry import MODELS
from mmdet.models.detectors.oag_staged import OAGStaged
from mmdet.models.detectors.sonarbind_prism import SonarBindPrism
from mmdet.models.layers.transformer.deformable_detr_layers import DeformableDetrTransformerEncoder
from mmdet.models.layers.transformer.oag_rgb_prompt_decoder import OAGRGBPromptDecoder
from mmdet.models.utils.sonar_rgb_prompt import SonarRGBPrompt


@MODELS.register_module()
class OAGNoTextAblation(OAGStaged):
    def __init__(self, closed_set_classes=6, **kwargs):
        # OAGRGBPrompt's constructor checks these flags. Its text-dependent
        # paths are all replaced below; _init_layers never constructs BERT.
        kwargs.update(use_text_branch=True, use_language_guided_query=True,
                      training_stage='fusion')
        super().__init__(**kwargs)
        if closed_set_classes != self.bbox_head.num_classes:
            raise ValueError('closed_set_classes must match bbox_head.num_classes')
        self.category_vectors = nn.Embedding(closed_set_classes, self.embed_dims)
        self.use_text_branch = False
        self.use_language_guided_query = False

    def _init_layers(self):
        from copy import deepcopy
        self._no_text_decoder_cfg = deepcopy(self.decoder)
        # Reuse backbone/projection/query initialization, skip language build.
        self.use_text_branch = False
        SonarBindPrism._init_layers(self)
        self.use_text_branch = True
        cfg = dict(self.encoder_cfg)
        cfg.pop('text_layer_cfg', None)
        cfg.pop('fusion_layer_cfg', None)
        self.encoder = DeformableDetrTransformerEncoder(**cfg)
        # The parent already turned decoder config into a module.
        decoder_cfg = self._no_text_decoder_cfg
        self.decoder = OAGRGBPromptDecoder(**decoder_cfg)
        for layer in self.decoder.layers:
            layer.rgb_prompt_attn.weight = self.decoder_sonar_attention_weight
            layer.cross_attn_text = None
            layer.use_text_cross_attn = False
            layer.norms = nn.ModuleList([layer.norms[0], layer.norms[2], layer.norms[3]])
        self.rgb_prompt_mapper = SonarRGBPrompt(self.embed_dims, self.prompt_temperature)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Baseline decoder has four norms; text-free decoder has three.
        # Detect baseline checkpoints by BERT keys (absent in our own saves).
        if any(k.startswith(prefix + 'language_model.') for k in state_dict):
            for i in range(self.decoder.num_layers):
                stem = prefix + f'decoder.layers.{i}.norms.'
                for suffix in ('weight', 'bias'):
                    for old, new in ((2, 1), (3, 2)):
                        key = stem + str(old) + '.' + suffix
                        if key in state_dict:
                            state_dict[stem + str(new) + '.' + suffix] = state_dict[key]
                    state_dict.pop(stem + '3.' + suffix, None)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward_encoder(self, feat, feat_mask, feat_pos, sonar_feat, sonar_mask,
                        sonar_pos, spatial_shapes, level_start_index, valid_ratios,
                        text_dict=None):
        memory = self.encoder(
            query=feat, query_pos=feat_pos, key_padding_mask=feat_mask,
            spatial_shapes=spatial_shapes, level_start_index=level_start_index,
            valid_ratios=valid_ratios)
        categories = self.category_vectors.weight[None].expand(feat.size(0), -1, -1)
        valid = torch.ones(categories.shape[:2], dtype=torch.bool, device=feat.device)
        return dict(memory=memory, memory_mask=feat_mask, spatial_shapes=spatial_shapes,
                    memory_text=categories, text_token_mask=valid,
                    memory_sonar=sonar_feat, memory_sonar_mask=sonar_mask)

    def _class_token_maps(self, samples, text_mask, dtype):
        count = self.category_vectors.num_embeddings
        maps = torch.eye(count, device=text_mask.device, dtype=dtype)
        return maps[None].expand(len(samples), -1, -1), text_mask

    @staticmethod
    def _head_inputs(inputs):
        return {k: v for k, v in inputs.items() if k not in
                ('memory_sonar', 'memory_sonar_mask', 'memory', 'memory_mask', 'spatial_shapes')}

    def loss(self, batch_inputs, batch_data_samples):
        width = self.bbox_head.max_text_len
        count = self.category_vectors.num_embeddings
        for sample in batch_data_samples:
            labels = sample.gt_instances.labels
            sample.gt_instances.positive_maps = F.one_hot(labels, width).float()
            sample.gt_instances.text_token_mask = torch.ones(
                (len(labels), count), device=labels.device, dtype=torch.bool)
        self._prompt_losses = None
        inputs = self.forward_transformer(self.extract_feat(batch_inputs), None, batch_data_samples)
        losses = self.bbox_head.loss(**self._head_inputs(inputs),
                                     batch_data_samples=batch_data_samples)
        losses.update(self._prompt_losses)
        self._prompt_losses = None
        return losses

    def predict(self, batch_inputs, batch_data_samples, rescale=True):
        count = self.category_vectors.num_embeddings
        for sample in batch_data_samples:
            sample.token_positive_map = {i + 1: [i] for i in range(count)}
        inputs = self.forward_transformer(self.extract_feat(batch_inputs), None, batch_data_samples)
        predictions = self.bbox_head.predict(**self._head_inputs(inputs),
            batch_data_samples=batch_data_samples, rescale=rescale)
        return self.add_pred_to_datasample(batch_data_samples, predictions)

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
        selection = pyramid  # Only mapped sonar evidence ranks RGB locations.
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

