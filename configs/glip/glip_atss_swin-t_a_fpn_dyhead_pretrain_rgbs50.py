_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

lang_model_name = '/media/fishyu/fish-14tb-2/YiFei/Grounding_DINO/mmdetection/bert-base-uncased'
pretrained = '/media/fishyu/fish-14tb-2/YiFei/Grounding_DINO/mmdetection/checkpoints/swin_tiny_patch4_window7_224.pth'  # noqa

model = dict(
    type='GLIP',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False,
        pad_size_divisor=32),
    backbone=dict(
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),
        with_cp=True,
        convert_weights=True,
        frozen_stages=-1,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        type='FPN',
        in_channels=[192, 384, 768],
        out_channels=256,
        start_level=0,
        relu_before_extra_convs=True,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='ATSSVLFusionHead',
        lang_model_name=lang_model_name,
        num_classes=6,
        in_channels=256,
        feat_channels=256,
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128],
            center_offset=0.5),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoderForGLIP',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
    ),
    language_model=dict(type='BertModel', name=lang_model_name),
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        # imdecode_backend='pillow',
        backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='GTBoxSubOne_GLIP'),
    dict(
        type='RandomChoiceResize',
        scales=[(1333, 480), (1333, 560), (1333, 640), (1333, 720),
                (1333, 800)],
        keep_ratio=True,
        resize_type='FixScaleResize',
        backend='pillow'),
    dict(type='RandomFlip_GLIP', prob=0.5),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1)),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction',
                   'custom_entities'))
]

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        backend_args=_base_.backend_args),
        # imdecode_backend='pillow'),
    dict(
        type='FixScaleResize',
        scale=(800, 1333),
        keep_ratio=True,
        backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'custom_entities'))
]

dataset_type = 'CocoDataset'
data_root = '/media/fishyu/fish-14tb-2/YiFei/Dataset/RGBS50_image/'

# RGBS50数据集类别定义
metainfo = {
    'classes': ('connected_polyhedron', 'fake_person', 'frustum',
                'iron_ball', 'octahedron', 'uuv'),
    'palette': [(119, 11, 32), (0, 0, 142), (0, 0, 230),
                (106, 0, 228), (0, 60, 100), (0, 80, 100)]
}

# 训练数据集
train_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    metainfo=metainfo,
    ann_file='instances_train_no_cb.json',
    data_prefix=dict(img='train_images/'),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=train_pipeline,
    return_classes=True,
    backend_args=None)

# 验证数据集
val_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    metainfo=metainfo,
    ann_file='instances_val_no_cb.json',
    data_prefix=dict(img='val_images/'),
    test_mode=True,
    pipeline=test_pipeline,
    return_classes=True,
    backend_args=None)

train_dataloader = dict(
    _delete_=True,
    batch_size=24,  # 双图像输入，减小batch_size
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=train_dataset)

val_dataloader = dict(
    _delete_=True,
    batch_size=24,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=val_dataset)

test_dataloader = val_dataloader

val_evaluator = dict(                                                                                                                                                                
      type='CocoMetric',                                                                                                                                                               
      ann_file=data_root + 'instances_val_no_cb.json',                                                                                                                                 
      metric='bbox',                                                                                                                                                                   
      backend_args=None)                                                                                                                                                               
test_evaluator = val_evaluator

# We did not adopt the official 24e optimizer strategy
# because the results indicate that the current strategy is superior.
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }),
    clip_grad=None)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=20,
        by_epoch=True,
        milestones=[8, 11],  # 在第8和11个epoch降低学习率
        gamma=0.1)
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=20, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')