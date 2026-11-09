_base_ = ['../oag/oag_double_resnet50_rgbs50_rgb_prompt_contrastive.py']
custom_imports = dict(
    imports=['mmdet.datasets', 'mmdet.models', 'mmdet.models.detectors.oag_staged'],
    allow_failed_imports=False)
# Keep the DN embedding shape identical across all three stages (4/10/6 classes).
# Detection classes come from each dataset's text; 10 is the shared DN capacity.
model = dict(type='OAGStaged', bbox_head=dict(num_classes=10),
             backbone=dict(frozen_stages=-1),
             sonar_backbone=dict(frozen_stages=-1), encoder=dict(num_cp=0))
max_epochs = 12
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', by_epoch=True, begin=0, end=12,
         milestones=[8, 11], gamma=0.1)]
# Stages rebuild the optimizer; do not use --resume when switching stages.
resume = False
model_wrapper_cfg = dict(type='MMDistributedDataParallel',
                         find_unused_parameters=True, static_graph=False)
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1,
                                    max_keep_ckpts=15))
optim_wrapper = dict(
    _delete_=True, type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=1e-4),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={
        'backbone': dict(lr_mult=0.1),
        'sonar_backbone': dict(lr_mult=0.1),
        'language_model': dict(lr_mult=0.01)}))
