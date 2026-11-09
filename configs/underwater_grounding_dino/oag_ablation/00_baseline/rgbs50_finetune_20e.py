# Exact reference architecture, initialization, data and 20-epoch schedule.
_base_ = ['../../oag_train/oag_double_resnet50_single_fusion_rgbs50_finetune_20e.py']
work_dir = './work_dirs/oag_ablation/00_baseline/rgbs50_finetune_20e'
randomness = dict(seed=42, deterministic=False)
