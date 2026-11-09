"""Optional decoder attention over independent sonar coordinates."""
import torch.nn as nn
from .sonarbind_layers import (SonarBindTransformerDecoder,
                               SonarBindTransformerDecoderLayer)


class OAGSonarDecoderLayer(SonarBindTransformerDecoderLayer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sonar_attention_weight = 0.5
        self.sonar_mode = 'feature'
        self.cross_attn_sonar = nn.MultiheadAttention(
            self.embed_dims, self.self_attn_cfg['num_heads'],
            dropout=0.0, batch_first=True)
        self.sonar_norm = nn.LayerNorm(self.embed_dims)

    def forward(self, query, query_pos=None, memory_sonar=None,
                sonar_pos=None, memory_sonar_mask=None,
                sonar_selection_weights=None, **kwargs):
        query = super().forward(query, query_pos=query_pos, **kwargs)
        if self.sonar_mode == 'position_only':
            if sonar_pos is None or sonar_selection_weights is None:
                raise ValueError('Position-only attention requires positions and weights')
            # A differentiable log prior over sonar locations. No sonar feature
            # content enters K/V and no RGB-to-sonar coordinate mapping is used.
            prior = sonar_selection_weights.to(query.dtype).clamp_min(1e-6).log()
            if memory_sonar_mask is not None:
                prior = prior.masked_fill(memory_sonar_mask, float('-inf'))
            bias = prior[:, None, :].expand(-1, query.size(1), -1)
            bias = bias.repeat_interleave(self.cross_attn_sonar.num_heads, dim=0)
            context, _ = self.cross_attn_sonar(
                query if query_pos is None else query + query_pos,
                sonar_pos, sonar_pos, attn_mask=bias, need_weights=False)
            # Normalize only the update: weight=0 is an exact identity.
            return query + self.sonar_attention_weight * self.sonar_norm(context)
        if memory_sonar is None or sonar_pos is None:
            raise ValueError('Sonar decoder requires selected features and positions')
        context, _ = self.cross_attn_sonar(
            query if query_pos is None else query + query_pos,
            memory_sonar + sonar_pos, memory_sonar,
            key_padding_mask=memory_sonar_mask, need_weights=False)
        return self.sonar_norm(query + self.sonar_attention_weight * context)


class OAGSonarDecoder(SonarBindTransformerDecoder):
    def _init_layers(self):
        super()._init_layers()
        self.layers = nn.ModuleList([
            OAGSonarDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)])
