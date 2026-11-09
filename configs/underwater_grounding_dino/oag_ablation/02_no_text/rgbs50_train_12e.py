# Rebuild the RGBS50 stage using this ablated architecture from epoch 1.
_base_ = ['./rgbs50_finetune_20e.py']
load_from = None
resume = False
max_epochs = 12
train_cfg = dict(max_epochs=max_epochs, val_interval=1)
optim_wrapper = dict(optimizer=dict(lr=1e-4))
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', by_epoch=True, begin=0, end=max_epochs,
         milestones=[8, 11], gamma=0.1)]
work_dir = './work_dirs/oag_ablation/02_no_text/rgbs50_resnet_only_train_12e'
