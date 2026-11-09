"""Affine-residual sonar position prior for OAG query selection only."""

from mmdet.registry import MODELS

from .oag_affine import OAGAffine


@MODELS.register_module()
class OAGAffineQueryOnly(OAGAffine):
    """Use mapped sonar positions to rank proposals, keeping the OAG decoder.

    ``OAGAffine`` also passes sonar features and positions to its decoder.
    Remove those auxiliary arguments here so that the original OAG decoder
    receives exactly the same inputs as it does without sonar cross-attention.
    """

    def __init__(self, **kwargs):
        if kwargs.pop('decoder_sonar_cross_attn', False):
            raise ValueError('OAGAffineQueryOnly requires the original decoder')
        super().__init__(decoder_sonar_cross_attn=False, **kwargs)

    def pre_decoder(self, *args, **kwargs):
        decoder_inputs, head_inputs = super().pre_decoder(*args, **kwargs)
        for key in ('sonar_pos', 'sonar_selection_weights',
                    'memory_sonar', 'memory_sonar_mask'):
            decoder_inputs.pop(key, None)
        return decoder_inputs, head_inputs
