# Contrastive-loss-only ablation; architecture and other losses unchanged.
_base_ = ['../../../oag_train/oag_double_resnet50_single_fusion_rgbs20_finetune_12e.py']
model = dict(
    enable_legacy_contrastive=True,
    loss_sonar_image_weight=0.25,
    loss_sonar_text_weight=0.25)
randomness = dict(seed=42, deterministic=False)
resume = False
work_dir = './work_dirs/oag_ablation/contrastive_loss/02_add_sonar_rgb/rgbs20_finetune_12e'
load_from = None
