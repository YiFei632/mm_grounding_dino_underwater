_base_ = ['../../../oag_train/oag_double_resnet50_single_fusion_rgbs20_finetune_12e.py']
custom_imports = dict(
    imports=['mmdet.datasets', 'mmdet.models',
             'configs.underwater_grounding_dino.oag_ablation.query_selector.adapter'],
    allow_failed_imports=False)
model = dict(type='OAGQuerySelectorAblation',
             query_selector_mode='fixed', num_queries=900)
randomness = dict(seed=42, deterministic=False)
resume = False
work_dir = './work_dirs/oag_ablation/query_selector/01_fixed_queries/rgbs20_finetune_12e'
load_from = None
