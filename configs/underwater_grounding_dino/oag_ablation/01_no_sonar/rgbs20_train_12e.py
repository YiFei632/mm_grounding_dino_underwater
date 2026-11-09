# Independent RGBS20 experiment; no transfer from RGBS50 or OAG checkpoints.
_base_ = ['./rgbs20_finetune_12e.py']
custom_imports = dict(
    imports=['mmdet.datasets', 'mmdet.models',
             'configs.underwater_grounding_dino.oag_ablation.resnet_only_init'],
    allow_failed_imports=False)
model = dict(language_model=dict(type='RandomInitAblationBert'))
load_from = None
resume = False
max_epochs = 12
train_cfg = dict(max_epochs=max_epochs, val_interval=1)
optim_wrapper = dict(optimizer=dict(lr=1e-4))
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', by_epoch=True, begin=0, end=max_epochs,
         milestones=[8, 11], gamma=0.1)]
work_dir = './work_dirs/oag_ablation/01_no_sonar/rgbs20_resnet_only_train_12e'
