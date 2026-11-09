_base_ = ['../../../oag_train/oag_double_resnet50_single_fusion_rgbs20_zeroshot.py']
custom_imports = dict(
    imports=['mmdet.datasets', 'mmdet.models',
             'configs.underwater_grounding_dino.oag_ablation.query_selector.adapter'],
    allow_failed_imports=False)
model = dict(type='OAGQuerySelectorAblation',
             query_selector_mode='no_sonar_prior', num_queries=900)
randomness = dict(seed=42, deterministic=False)
resume = False
work_dir = './work_dirs/oag_ablation/query_selector/03_no_sonar_prior/rgbs20_zeroshot'
load_from = '/media/fishyu/fish-14tb-2/YiFei/Grounding_DINO/mmdetection/work_dirs/oag_ablation/query_selector/03_no_sonar_prior/rgbs50_finetune_20e/epoch_12.pth'

# Preserve the requested one-epoch training protocol.
max_epochs = 1
train_cfg = dict(max_epochs=1, val_interval=1)
