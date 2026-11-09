_base_ = ['./oag_double_resnet50_rgbs50_rgb_prompt.py']

# Opt-in legacy SonarBind contrastive objectives. Both values are deliberately
# exposed here so ablations can change them without touching Python code.
model = dict(
    enable_legacy_contrastive=True,
    loss_sonar_image_weight=0,
    loss_sonar_text_weight=0.25,
)
max_epochs = 20
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=2)  # 每5个epoch验证一次
