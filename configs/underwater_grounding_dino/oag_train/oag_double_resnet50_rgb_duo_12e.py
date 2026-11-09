_base_ = ['./_base_staged.py']
model = dict(training_stage='rgb', pretrain_mode=False,
    sonar_position_weight=0., decoder_sonar_attention_weight=0.,
    enable_legacy_contrastive=False, loss_sonar_image_weight=0.,
    loss_sonar_text_weight=0., loss_rgb_prompt_weight=0.,
    loss_sonar_class_weight=1., loss_roi_text_weight=1.,bbox_head=dict(num_classes=4))
work_dir = './work_dirs/oag_train/rgb_duo'
load_from = '/media/fishyu/fish-14tb-2/YiFei/Grounding_DINO/mmdetection/work_dirs/oag_train/rgb_uxo/epoch_12.pth'
data_root = '/media/fishyu/fish-14tb-2/YiFei/Dataset/DUO/'
metainfo = dict(classes=('holothurian', 'echinus', 'scallop', 'starfish'))
train_pipeline = [dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomChoiceResize', scales=[(480, 1333), (640, 1333), (800, 1333)], keep_ratio=True), dict(type='RandomFlip', prob=0.5),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='PackDetInputs')]
test_pipeline = [dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(800, 1333), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True), dict(type='PackDetInputs')]
train_dataloader = dict(_delete_=True, batch_size=4, num_workers=4,
    persistent_workers=True, sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(type='CocoDataset', data_root=data_root, metainfo=metainfo,
        ann_file='annotations/instances_train.json', data_prefix=dict(img='images/train/'),
        filter_cfg=dict(filter_empty_gt=False), return_classes=True,
        test_mode=False, pipeline=train_pipeline))
val_dataloader = dict(_delete_=True, batch_size=4, num_workers=4,
    persistent_workers=True, sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type='CocoDataset', data_root=data_root, metainfo=metainfo,
        ann_file='annotations/instances_test.json', data_prefix=dict(img='images/test/'),
        filter_cfg=dict(filter_empty_gt=False), return_classes=True,
        test_mode=True, pipeline=test_pipeline))
val_evaluator = dict(_delete_=True, type='CocoMetric',
    ann_file=data_root + 'annotations/instances_test.json', metric='bbox')
test_dataloader = val_dataloader
test_evaluator = val_evaluator
