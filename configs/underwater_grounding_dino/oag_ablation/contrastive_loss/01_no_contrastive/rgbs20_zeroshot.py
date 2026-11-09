# Contrastive-loss-only ablation; architecture and other losses unchanged.
_base_ = ['../../../oag_train/oag_double_resnet50_single_fusion_rgbs20_zeroshot.py']
model = dict(
    enable_legacy_contrastive=False,
    loss_sonar_image_weight=0.0,
    loss_sonar_text_weight=0.0)
randomness = dict(seed=42, deterministic=False)
resume = False
work_dir = './work_dirs/oag_ablation/contrastive_loss/01_no_contrastive/rgbs20_zeroshot'
load_from = '/media/fishyu/fish-14tb-2/YiFei/Grounding_DINO/mmdetection/work_dirs/oag_ablation/contrastive_loss/01_no_contrastive/rgbs50_finetune_20e/epoch_12.pth'

# Preserve the intentional one-epoch training protocol.
max_epochs = 1
train_cfg = dict(max_epochs=1, val_interval=1)
