"""Query-selector-only overrides; downstream branches and losses retained."""
import math
import torch
import torch.nn.functional as F
from mmdet.registry import MODELS
from mmdet.models.detectors.oag_staged import OAGStaged


@MODELS.register_module()
class OAGQuerySelectorAblation(OAGStaged):
    def __init__(self, query_selector_mode, **kwargs):
        if query_selector_mode not in ('fixed', 'no_text_prior', 'no_sonar_prior'):
            raise ValueError('Unknown query selector ablation')
        self.query_selector_mode = query_selector_mode
        super().__init__(**kwargs)
        if self.training_stage != 'fusion':
            raise ValueError('Query selector experiments require fusion stage')
        side = math.ceil(math.sqrt(self.num_queries))
        axis = (torch.arange(side, dtype=torch.float32) + .5) / side
        y, x = torch.meshgrid(axis, axis, indexing='ij')
        centers = torch.stack((x.flatten(), y.flatten()), -1)[:self.num_queries]
        boxes = torch.cat((centers, torch.full_like(centers, .05)), -1)
        self.register_buffer('fixed_query_boxes', boxes, persistent=False)

    def _selector_indices(self, logits, pyramid, proposals, memory_mask):
        invalid = memory_mask | ~torch.isfinite(proposals).all(-1)
        if ((~invalid).sum(-1) < self.num_queries).any():
            raise ValueError('Fewer valid RGB proposals than num_queries')
        if self.query_selector_mode == 'fixed':
            # Content-independent samples for encoder auxiliary loss ONLY.
            # Decoder uses fixed boxes, never these selected/regressed boxes.
            indices = []
            for valid in ~invalid:
                locations = valid.nonzero(as_tuple=False).flatten()
                offsets = torch.linspace(0, len(locations) - 1,
                                         self.num_queries, device=locations.device).long()
                indices.append(locations[offsets])
            return torch.stack(indices)
        if self.query_selector_mode == 'no_text_prior':
            selection = self.sonar_position_weight * pyramid
        else:
            selection = logits
        scores = selection.max(-1).values.masked_fill(invalid, float('-inf'))
        return scores.topk(self.num_queries, dim=1).indices

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
        indices = self._selector_indices(logits, pyramid, proposals, memory_mask)
        top_logits = logits.gather(1, indices[..., None].expand(-1, -1, logits.size(-1)))
        top_coords = coords.gather(1, indices[..., None].expand(-1, -1, 4))
        query = self.query_embedding.weight[None].expand(bs, -1, -1)
        reference = top_coords.detach()
        if self.query_selector_mode == 'fixed':
            reference = torch.logit(self.fixed_query_boxes.to(query))[None].expand(bs, -1, -1)
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

