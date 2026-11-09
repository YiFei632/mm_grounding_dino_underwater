# Preserve the single-fusion model, losses, datasets and batch sizes exactly.
_base_ = ['./oag_double_resnet50_single_fusion_rgbs50_12e.py']

work_dir = './work_dirs/oag_train/single_fusion_rgbs50_finetune_20e'
load_from = '/media/fishyu/fish-14tb-2/YiFei/Grounding_DINO/mmdetection/work_dirs/oag_train/single_fusion_rgbs50/epoch_12.pth'
# Start a fresh 20-epoch optimization schedule from the trained weights.
resume = False

max_epochs = 20
train_cfg = dict(max_epochs=max_epochs, val_interval=1)
optim_wrapper = dict(optimizer=dict(lr=5e-5))
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', by_epoch=True, begin=0, end=max_epochs,
         milestones=[12, 18], gamma=0.1),
]
