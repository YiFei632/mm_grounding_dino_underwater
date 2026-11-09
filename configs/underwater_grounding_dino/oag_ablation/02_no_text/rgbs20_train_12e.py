_base_ = ['../../oag_train/oag_double_resnet50_single_fusion_rgbs20_finetune_12e.py']
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

# Transfer from this ablation's RGBS50 fine-tuning checkpoint.
load_from = None
resume = False
work_dir = './work_dirs/oag_ablation/02_no_text/rgbs20_resnet_only_train_12e'
