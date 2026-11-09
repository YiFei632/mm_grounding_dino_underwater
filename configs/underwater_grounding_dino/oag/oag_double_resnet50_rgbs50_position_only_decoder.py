# Sonar content affects attention only through the selector's spatial prior.
# K/V contain positional encodings, never sonar appearance features.
_base_ = ['./oag_double_resnet50_rgbs50_sonar_decoder.py']

model = dict(
    decoder_sonar_mode='position_only',
    decoder_sonar_attention_weight=0.1,
)
