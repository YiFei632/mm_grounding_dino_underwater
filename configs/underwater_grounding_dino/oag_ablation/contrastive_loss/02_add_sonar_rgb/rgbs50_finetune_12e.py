# Contrastive-loss-only ablation; architecture and other losses unchanged.
_base_ = ['../../../oag_train/oag_double_resnet50_single_fusion_rgbs50_finetune_20e.py']
model = dict(
    enable_legacy_contrastive=True,
    loss_sonar_image_weight=0.25,
    loss_sonar_text_weight=0.25)
randomness = dict(seed=42, deterministic=False)
resume = False
work_dir = './work_dirs/oag_ablation/contrastive_loss/02_add_sonar_rgb/rgbs50_finetune_20e'
max_epochs = 12
train_cfg = dict(max_epochs=max_epochs, val_interval=1)
optim_wrapper = dict(optimizer=dict(lr=5e-5))
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', by_epoch=True, begin=0, end=max_epochs,
         milestones=[8, 11], gamma=0.1),
]
load_from = None