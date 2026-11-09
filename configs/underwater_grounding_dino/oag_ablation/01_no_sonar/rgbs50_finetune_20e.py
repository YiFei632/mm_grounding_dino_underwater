_base_ = ['../00_baseline/rgbs50_finetune_20e.py']
# Build the existing RGB+language detector. No sonar modules are instantiated.
model = dict(
    _delete_=True,
    type='GroundingDINO',
    num_queries=_base_.model.num_queries,
    with_box_refine=True,
    as_two_stage=True,
    data_preprocessor=_base_.model.data_preprocessor,
    language_model=_base_.model.language_model,
    backbone=_base_.model.backbone,
    neck=_base_.model.neck,
    encoder=dict(
        num_layers=_base_.model.encoder.num_layers,
        num_cp=0,
        layer_cfg=_base_.model.encoder.layer_cfg,
        text_layer_cfg=_base_.model.encoder.text_layer_cfg,
        fusion_layer_cfg=_base_.model.encoder.fusion_layer_cfg),
    decoder=_base_.model.decoder,
    positional_encoding=_base_.model.positional_encoding,
    bbox_head=_base_.model.bbox_head,
    dn_cfg=_base_.model.dn_cfg,
    train_cfg=_base_.model.train_cfg,
    test_cfg=_base_.model.test_cfg)

# Keep RGB augmentation and the exact same RGBS50 image/annotation split.
train_pipeline = [step for step in _base_.train_pipeline
                  if step['type'] not in ('LoadSonarImage', 'LoadSonarAnnotations')]
test_pipeline = [step for step in _base_.test_pipeline
                 if step['type'] not in ('LoadSonarImage', 'LoadSonarAnnotations')]
for step in train_pipeline + test_pipeline:
    if step['type'] == 'PackDetInputs':
        step['meta_keys'] = tuple(key for key in step['meta_keys']
                                  if not key.startswith('sonar_'))
train_dataset = dict(pipeline=train_pipeline)
val_dataset = dict(pipeline=test_pipeline)
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
work_dir = './work_dirs/oag_ablation/01_no_sonar/rgbs50_finetune_20e'
