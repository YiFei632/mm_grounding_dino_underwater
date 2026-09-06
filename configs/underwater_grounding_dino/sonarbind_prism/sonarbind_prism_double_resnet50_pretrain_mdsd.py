_base_ = [
    '../../_base_/datasets/coco_detection.py',
    '../../_base_/schedules/schedule_1x.py', '../../_base_/default_runtime.py'
]

# Custom imports to ensure all transforms are registered
custom_imports = dict(
    imports=['mmdet.datasets', 'mmdet.models'],
    allow_failed_imports=False)

lang_model_name = '/media/fishyu/fish-14tb-2/YiFei/Grounding_DINO/mmdetection/bert-base-uncased'

model = dict(
    type='SonarBindPrismPretrain',
    num_queries=900,
    with_box_refine=True,
    as_two_stage=True,

    # ========== 预训练模式参数 ==========
    pretrain_mode=True,  # 开启预训练模式
    pretrain_modality='sonar',  # 训练声呐分支
    use_text_branch=True,
    pretrain_text_templates=[
        "a photo of a {}",
        "a {} in the underwater scene",
        "an underwater {}",
        "a {} in the ocean",
    ],
    pretrain_loss_weight=1.0,
    pretrain_roi_output_size=7,
    pretrain_roi_spatial_scale=0.125,  # 1/8, 根据特征图stride
    # ===================================

    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=False,
    ),
    language_model=dict(
        type='BertModel',
        name=lang_model_name,
        max_tokens=256,
        pad_to_max=False,
        use_sub_sentence_represent=True,
        special_tokens_list=['[CLS]', '[SEP]', '.', '?'],
        add_pooling_layer=False,
    ),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),  # 只要最后一层特征
        frozen_stages=4,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch'
    ),
    sonar_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3,),  # 只要最后一层特征
        frozen_stages=1,  
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'
    ),
    # ========== 移除FFT Band0滤波（预训练不用声纳） ==========
    fft_band0_filter=None,
    # ============================================
    # Contrastive loss configuration (预训练模式不使用)
    loss_sonar_image_weight=0.0,  # 预训练模式不使用
    loss_sonar_text_weight=0.0,   # 预训练模式不使用
    contrastive_temperature=0.07,  # Temperature parameter (CLIP default)
    learnable_temperature=False,   # Whether temperature is learnable
    neck=dict(
        type='ChannelMapper',
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        bias=True,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    encoder=dict(
        type='SonarBindTransformerEncoder',
        num_layers=6,
        num_cp=0,  # 预训练模式禁用checkpoint避免DDP错误
        # visual layer config
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=4, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0)),
        # text layer config
        text_layer_cfg=dict(
            self_attn_cfg=dict(num_heads=4, embed_dims=256, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=1024, ffn_drop=0.0)),
        # fusion layer config
        fusion_layer_cfg=dict(
            v_dim=256,
            l_dim=256,
            embed_dim=1024,
            num_heads=4,
            init_values=1e-4),
    ),
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            # query self attention layer
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            # cross attention layer query to text
            cross_attn_text_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            # cross attention layer query to image
            cross_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0)),
        post_norm_cfg=None),
    positional_encoding=dict(
        num_feats=128, normalize=True, offset=0.0, temperature=20),
    bbox_head=dict(
        type='GroundingDINOHead',
        num_classes=10,  # UTDAC2020有4个类别
        sync_cls_avg_factor=True,
        contrastive_cfg=dict(max_text_len=256, log_scale='auto', bias=True),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0)),
    dn_cfg=dict(  # TODO: Move to model.train_cfg ?
        label_noise_scale=0.5,
        box_noise_scale=1.0,  # 0.4 for DN-DETR
        group_cfg=dict(dynamic=True, num_groups=None,
                       num_dn_queries=100)),  # TODO: half num_dn_queries
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='BinaryFocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(max_per_img=300))

# dataset settings
# UTDAC2020数据集根目录
data_root = '/media/fishyu/fish-14tb-2/YiFei/Dataset/MDSD_COCO/'

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    # This dataset's primary image is sonar and has its own COCO boxes.
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(400, 1333), (500, 1333), (600, 1333)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction'))
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None, imdecode_backend='pillow'),
    # This dataset's primary image is sonar and has its own COCO boxes.
    dict(type='FixScaleResize', scale=(800, 1333), keep_ratio=True, backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

dataset_type = 'CocoDataset'

# UTDAC2020数据集类别定义
metainfo = {
    'classes': ('can', 'bottle', 'drink-carton', 'chain', 'propeller', 'tire', 'hook', 'valve', 'shampoo-bottle', 'standing-bottle'),
    'palette': [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (110, 20, 60), (229, 11, 32), (0, 142, 0), (0, 230, 0), (0, 20, 60), (9, 11, 32)]
}

# 训练数据集
train_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    metainfo=metainfo,
    ann_file='annotations/instances_train2017.json',
    data_prefix=dict(img='train2017/'),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=train_pipeline,
    return_classes=True,
    backend_args=None)

# 验证数据集
val_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    metainfo=metainfo,
    ann_file='annotations/instances_val2017.json',
    data_prefix=dict(img='val2017/'),
    pipeline=test_pipeline,
    return_classes=True,
    backend_args=None)

train_dataloader = dict(
    _delete_=True,
    batch_size=4,  # 预训练可以用较小的batch size
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=train_dataset)

val_dataloader = dict(
    _delete_=True,
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=val_dataset)

test_dataloader = val_dataloader

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.0),  # RGB Backbone完全冻结
            'language_model': dict(lr_mult=0.1),  # Language Model用小学习率
            'sonar_backbone': dict(lr_mult=0.2),  # Sonar小学习率
            'decoder': dict(lr_mult=0.0),  # Decoder冻结（预训练不用）
            'bbox_head': dict(lr_mult=0.0),  # BBox Head冻结（预训练不用）
        }))

# learning policy
max_epochs = 12  # 预训练12个epoch
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[8, 11],  # 在第8和11个epoch降低学习率
        gamma=0.1)
]

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=6)  # 每2个epoch验证一次

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 评估指标（预训练模式可选，主要监控loss）
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val2017.json',
    metric='bbox',
    format_only=False,
    backend_args=None)

test_evaluator = val_evaluator

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (1 GPU) x (4 samples per GPU)
auto_scale_lr = dict(base_batch_size=4, enable=False)  # 关闭自动学习率缩放

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=2, max_keep_ckpts=3, save_best='auto'),
    logger=dict(type='LoggerHook', interval=50))

model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=True,
    static_graph=False  # 预训练模式使用静态图避免DDP错误
)

load_from = '/media/fishyu/fish-14tb-2/YiFei/Grounding_DINO/mmdetection/work_dirs/sonarbind_prism_double_resnet50_pretrain_uatd/epoch_12.pth'
