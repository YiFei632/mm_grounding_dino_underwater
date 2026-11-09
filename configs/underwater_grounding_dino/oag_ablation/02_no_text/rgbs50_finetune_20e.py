_base_ = ['../00_baseline/rgbs50_finetune_20e.py']
custom_imports = dict(
    imports=['mmdet.datasets', 'mmdet.models',
             'configs.underwater_grounding_dino.oag_ablation.no_text_adapter'],
    allow_failed_imports=False)
model = dict(
    type='OAGNoTextAblation',
    # Constructor compatibility only: adapter never constructs BERT.
    language_model=None,
    closed_set_classes=6,
    enable_legacy_contrastive=False,
    loss_sonar_image_weight=0.,
    loss_sonar_text_weight=0.)
work_dir = './work_dirs/oag_ablation/02_no_text/rgbs50_finetune_20e'
