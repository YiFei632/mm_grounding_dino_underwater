"""Position-only RGB-grid attention conditioned on mapped sonar evidence."""
import torch
from torch import nn
import torch.nn.functional as F

from .sonarbind_layers import (SonarBindTransformerDecoder,
                               SonarBindTransformerDecoderLayer)


class RGBPromptAttention(nn.Module):
    def __init__(self, embed_dims, num_heads, weight=0.1):
        super().__init__()
        self.weight = weight
        self.attn = nn.MultiheadAttention(embed_dims, num_heads, batch_first=True)
        self.query_class = nn.Linear(embed_dims, embed_dims)
        self.text_class = nn.Linear(embed_dims, embed_dims)
        self.norm = nn.LayerNorm(embed_dims)

    def forward(self, query, query_pos, positions, prompt, classes, class_valid,
                padding_mask):
        q = query if query_pos is None else query + query_pos
        scores = (F.normalize(self.query_class(query).float(), dim=-1) @
                  F.normalize(self.text_class(classes).float(), dim=-1).transpose(1, 2)) / 0.07
        scores = scores.masked_fill(~class_valid[:, None], float('-inf'))
        mixture = scores.softmax(-1)
        prior = mixture @ prompt.float().transpose(1, 2)
        prior = prior.masked_fill(padding_mask[:, None], 0)
        gate = prior.amax(-1, keepdim=True)
        bias = prior.clamp_min(1e-6).log().masked_fill(
            padding_mask[:, None], float('-inf'))
        # A zero value null slot makes even an entirely padded grid finite.
        # It is only enabled for samples without any valid grid position.
        empty = padding_mask.all(-1)
        null_bias = torch.where(empty, 0., float('-inf'))[:, None, None]
        bias = torch.cat([bias, null_bias.expand(-1, query.size(1), 1)], -1)
        positions = torch.cat([positions, positions.new_zeros(positions.size(0), 1,
                                                           positions.size(2))], 1)
        bias = bias.to(query.dtype).repeat_interleave(self.attn.num_heads, dim=0)
        context, _ = self.attn(q, positions.to(q.dtype), positions.to(q.dtype),
                               attn_mask=bias, need_weights=False)
        return query + self.weight * gate.to(query.dtype) * self.norm(context)


class OAGRGBPromptDecoderLayer(SonarBindTransformerDecoderLayer):
    def __init__(self, prompt_weight=0.1, **kwargs):
        super().__init__(**kwargs)
        self.rgb_prompt_attn = RGBPromptAttention(
            self.embed_dims, self.self_attn_cfg['num_heads'], prompt_weight)

    def forward(self, query, query_pos=None, rgb_prompt=None,
                rgb_prompt_pos=None, rgb_prompt_mask=None,
                prompt_classes=None, prompt_class_valid=None, **kwargs):
        query = super().forward(query, query_pos=query_pos, **kwargs)
        if self.rgb_prompt_attn.weight == 0:
            return query
        return self.rgb_prompt_attn(query, query_pos, rgb_prompt_pos, rgb_prompt,
                                    prompt_classes, prompt_class_valid, rgb_prompt_mask)


class OAGRGBPromptDecoder(SonarBindTransformerDecoder):
    def _init_layers(self):
        super()._init_layers()
        self.layers = nn.ModuleList([
            OAGRGBPromptDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)])
