# Configuration for training LGANet backbone through SonarBindLGANet detector
# This config extends the base 6-epoch config to 50 epochs for thorough backbone training
# After training, extract the sonar_backbone weights using tools/extract_lganet_from_sonarbind.py

_base_ = './sonarbind_resnet50_lganet_pretrain_rgbs50_6epoch.py'

# Increase training epochs for better backbone learning
max_epochs = 10

# Adjust learning rate schedule for longer training
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[30, 40],  # Decay at 60% and 80% of training
        gamma=0.1)
]

# Update training loop
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=5  # Validate every 5 epochs
)

# Optional: Adjust sonar_backbone learning rate to focus on backbone training
# Uncomment to give sonar_backbone higher learning rate
# optim_wrapper = dict(
#     paramwise_cfg=dict(
#         custom_keys={
#             'absolute_pos_embed': dict(decay_mult=0.),
#             'backbone': dict(lr_mult=0.1),
#             'language_model': dict(lr_mult=0.1),
#             'sonar_backbone': dict(lr_mult=1.0),  # Full learning rate for sonar backbone
#         }))

# Optional: Freeze RGB backbone to focus training on sonar backbone
# Uncomment to freeze RGB backbone
# model = dict(
#     backbone=dict(
#         frozen_stages=4,  # Freeze all stages of RGB backbone
#     ),
# )
