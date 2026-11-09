_base_ = ['./_base_staged.py']
model = dict(training_stage='sonar', pretrain_mode=False,
    sonar_position_weight=0., decoder_sonar_attention_weight=0.,
    enable_legacy_contrastive=False, loss_sonar_image_weight=0.,
    loss_sonar_text_weight=0., loss_rgb_prompt_weight=0.,
    loss_sonar_class_weight=1., loss_roi_text_weight=1.,
    bbox_head=dict(num_classes=10))
work_dir = './work_dirs/oag_train/sonar_mdsd_coco'
# Change this path if RGB training uses another work_dir/checkpoint.
load_from = '/media/fishyu/fish-14tb-2/YiFei/Grounding_DINO/mmdetection/work_dirs/oag_train/sonar_uatd_coco/epoch_12.pth'
data_root = '/media/fishyu/fish-14tb-2/YiFei/Dataset/MDSD_COCO/'
metainfo = dict(classes=('can', 'bottle', 'drink-carton', 'chain', 'propeller', 'tire', 'hook', 'valve', 'shampoo-bottle', 'standing-bottle'))
train_pipeline = [dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(448, 448), keep_ratio=False), dict(type='RandomFlip', prob=0.5),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='PackDetInputs')]
test_pipeline = [dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(800, 1333), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True), dict(type='PackDetInputs')]
train_dataloader = dict(_delete_=True, batch_size=4, num_workers=4,
    persistent_workers=True, sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(type='CocoDataset', data_root=data_root, metainfo=metainfo,
        ann_file='annotations/instances_train2017.json', data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=False), return_classes=True,
        test_mode=False, pipeline=train_pipeline))
val_dataloader = dict(_delete_=True, batch_size=4, num_workers=4,
    persistent_workers=True, sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type='CocoDataset', data_root=data_root, metainfo=metainfo,
        ann_file='annotations/instances_val2017.json', data_prefix=dict(img='val2017/'),
        filter_cfg=dict(filter_empty_gt=False), return_classes=True,
        test_mode=True, pipeline=test_pipeline))
# Sonar stage learns heatmaps/ROI semantics, not a detection decoder.
# Evaluate downstream detection after fusion training; no meaningless sonar bbox AP.
val_dataloader = None
val_cfg = None
val_evaluator = None
test_dataloader = None
test_cfg = None
test_evaluator = None
