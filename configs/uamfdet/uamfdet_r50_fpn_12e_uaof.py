_base_ = ['../_base_/default_runtime.py']
custom_imports = dict(imports=['mmdet.models.fusion', 'mmdet.models.detectors.uamfdet',
                               'mmdet.datasets.uamfdet', 'mmdet.datasets.transforms.loading_uamfdet'],
                      allow_failed_imports=False)

model = dict(
    type='UAMFDet',
    data_preprocessor=dict(type='DetDataPreprocessor', mean=[123.675,116.28,103.53,123.675,116.28,103.53],
                           std=[58.395,57.12,57.375,58.395,57.12,57.375], bgr_to_rgb=False),
    backbone=dict(type='ResNet', depth=50, num_stages=4, out_indices=(0,1,2,3),
                  frozen_stages=1, norm_cfg=dict(type='BN', requires_grad=True),
                  norm_eval=True, style='pytorch', init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(type='FPN', in_channels=[256,512,1024,2048], out_channels=256, num_outs=5),
    fusion=dict(mdsff=dict(type='MDSFF', channels=256), mifm=dict(type='MIFM', channels=256)),
    rpn_head=dict(type='RPNHead', in_channels=256, feat_channels=256,
                  anchor_generator=dict(type='AnchorGenerator', scales=[8], ratios=[0.5,1.0,2.0], strides=[4,8,16,32,64]),
                  bbox_coder=dict(type='DeltaXYWHBBoxCoder', target_means=[0,0,0,0], target_stds=[1,1,1,1]),
                  loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                  loss_bbox=dict(type='SmoothL1Loss', beta=1.0/9, loss_weight=1.0)),
    roi_head=dict(type='StandardRoIHead', bbox_roi_extractor=dict(type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0), out_channels=256, featmap_strides=[4,8,16,32]),
        bbox_head=dict(type='Shared2FCBBoxHead', in_channels=256, fc_out_channels=1024, roi_feat_size=7,
                       num_classes=10, bbox_coder=dict(type='DeltaXYWHBBoxCoder', target_means=[0,0,0,0], target_stds=[0.1,0.1,0.2,0.2]),
                       reg_class_agnostic=False, loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                       loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))),
    train_cfg=dict(rpn=dict(assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.7, neg_iou_thr=0.3, min_pos_iou=0), sampler=dict(type='RandomSampler', num=256, pos_fraction=0.5, neg_pos_ub=-1, add_gt_as_proposals=False), allowed_border=-1, pos_weight=-1, debug=False),
                   rpn_proposal=dict(nms_pre=2000, max_per_img=512, nms=dict(type='nms', iou_threshold=0.7), min_bbox_size=0),
                   rcnn=dict(assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.5, neg_iou_thr=0.5, min_pos_iou=0, match_low_quality=False), sampler=dict(type='RandomSampler', num=512, pos_fraction=0.25, neg_pos_ub=-1, add_gt_as_proposals=True), pos_weight=-1, debug=False)),
    test_cfg=dict(rpn=dict(nms_pre=1000, max_per_img=1000, nms=dict(type='nms', iou_threshold=0.7), min_bbox_size=0), rcnn=dict(score_thr=0.05, nms=dict(type='nms', iou_threshold=0.5), max_per_img=100)))

train_pipeline=[dict(type='LoadUAMFDetImages'), dict(type='LoadAnnotations', with_bbox=True), dict(type='RandomFlip', prob=0.5), dict(type='PackDetInputs')]
test_pipeline=[dict(type='LoadUAMFDetImages'), dict(type='LoadAnnotations', with_bbox=True), dict(type='PackDetInputs')]
train_dataloader=dict(batch_size=4, num_workers=2, persistent_workers=True, sampler=dict(type='DefaultSampler', shuffle=True), dataset=dict(type='UAOFDataset', data_root='data/UAOF/', ann_file='annotations/train.json', data_prefix=dict(img='optical/', sonar='sonar/'), pipeline=train_pipeline))
val_dataloader=dict(batch_size=1, num_workers=2, persistent_workers=True, sampler=dict(type='DefaultSampler', shuffle=False), dataset=dict(type='UAOFDataset', data_root='data/UAOF/', ann_file='annotations/val.json', data_prefix=dict(img='optical/', sonar='sonar/'), pipeline=test_pipeline, test_mode=True))
test_dataloader=val_dataloader
train_cfg=dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
val_cfg=dict(type='ValLoop'); test_cfg=dict(type='TestLoop')
optim_wrapper=dict(optimizer=dict(type='Adam', lr=1e-4, weight_decay=0.1))
work_dir='work_dirs/uamfdet_r50_fpn_12e_uaof'
