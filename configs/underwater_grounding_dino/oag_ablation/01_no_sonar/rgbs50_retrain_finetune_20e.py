# Continue from this architecture's own RGBS50 12-epoch checkpoint.
_base_ = ['./rgbs50_finetune_20e.py']
load_from = './work_dirs/oag_ablation/01_no_sonar/rgbs50_train_12e/epoch_12.pth'
resume = False
work_dir = './work_dirs/oag_ablation/01_no_sonar/rgbs50_retrain_finetune_20e'
