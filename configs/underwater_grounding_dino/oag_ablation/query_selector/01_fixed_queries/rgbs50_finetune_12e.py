_base_ = ['../../../oag_train/oag_double_resnet50_single_fusion_rgbs50_finetune_20e.py']
custom_imports = dict(
    imports=['mmdet.datasets', 'mmdet.models',
             'configs.underwater_grounding_dino.oag_ablation.query_selector.adapter'],
    allow_failed_imports=False)
model = dict(type='OAGQuerySelectorAblation',
             query_selector_mode='fixed', num_queries=900)
randomness = dict(seed=42, deterministic=False)
resume = False
work_dir = './work_dirs/oag_ablation/query_selector/01_fixed_queries/rgbs50_finetune_20e'

max_epochs = 12
train_cfg = dict(max_epochs=max_epochs, val_interval=1)
optim_wrapper = dict(optimizer=dict(lr=5e-5))
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', by_epoch=True, begin=0, end=max_epochs,
         milestones=[8, 11], gamma=0.1),
]
