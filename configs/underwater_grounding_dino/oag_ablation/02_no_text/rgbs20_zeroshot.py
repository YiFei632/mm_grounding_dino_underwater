_base_ = ['../../oag_train/oag_double_resnet50_single_fusion_rgbs20_zeroshot.py']
custom_imports = dict(
    imports=['mmdet.datasets', 'mmdet.models',
             'configs.underwater_grounding_dino.oag_ablation.rgbs20_no_text_adapter'],
    allow_failed_imports=False)
model = dict(
    type='OAGNoTextRGBS20Ablation',
    language_model=None,
    closed_set_classes=4,
    enable_legacy_contrastive=False,
    loss_sonar_image_weight=0.,
    loss_sonar_text_weight=0.)

# Transfer from this ablation's RGBS20 fine-tuning checkpoint.
load_from = './work_dirs/oag_ablation/02_no_text/rgbs50_train_12e/epoch_12.pth'
resume = False
work_dir = './work_dirs/oag_ablation/02_no_text/rgbs20_zeroshot'

# The one-epoch training protocol is intentional.
max_epochs = 1
train_cfg = dict(max_epochs=1, val_interval=1)
