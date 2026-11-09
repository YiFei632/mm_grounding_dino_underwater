# Inherit the current OAG setup; enable sonar attention in every decoder layer.
_base_ = ['./oag_double_resnet50_rgbs50.py']
model = dict(
    decoder_sonar_cross_attn=True,
    decoder_sonar_attention_weight=0.5,
)
