_base_ = ['./oag_double_resnet50_rgbs50_position_only_decoder.py']

custom_imports = dict(
    imports=['mmdet.datasets', 'mmdet.models', 'mmdet.models.detectors.oag_rgb_prompt'],
    allow_failed_imports=False)

model = dict(
    type='OAGRGBPrompt',
    prompt_level=2,
    prompt_temperature=0.07,
    sonar_position_weight=0.5,
    decoder_sonar_attention_weight=0.1,
    loss_rgb_prompt_weight=1.0,
    loss_sonar_class_weight=1.0,
)
