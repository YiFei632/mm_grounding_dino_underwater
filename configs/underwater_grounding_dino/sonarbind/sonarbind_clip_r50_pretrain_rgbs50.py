_base_ = [
    '../../_base_/datasets/coco_detection.py',
    '../../_base_/schedules/schedule_1x.py',
    '../../_base_/default_runtime.py'
]

# Custom imports to ensure all transforms are registered
custom_imports = dict(
    imports=['mmdet.datasets', 'mmdet.models'],
    allow_failed_imports=False)

# ── 本地模型路径 ────────────────────────────────────────────────────────────
clip_model_path = '/media/fishyu/fish-14tb-2/YiFei/Grounding_DINO/mmdetection/clip-vit-large-patch14'

# ── 数据路径 ────────────────────────────────────────────────────────────────
data_root = '/media/fishyu/fish-14tb-2/YiFei/Dataset/RGBS50_image/'
sonar_data_root = '/media/fishyu/fish-14tb-2/YiFei/Dataset/RGBS50_sonar_corrected/'

# ── ImageNet 归一化参数（× 255，RGB 顺序）────────────────────────────────
# ImageNet 原始（RGB）：mean=[0.485, 0.456, 0.406]
#                       std =[0.229, 0.224, 0.225]
# data_preprocessor 先 bgr_to_rgb=True，再用下方 RGB 顺序的 mean/std
_imagenet_mean = [123.675, 116.28, 103.53]
_imagenet_std = [58.395, 57.12, 57.375]

model = dict(
    type='CLIPImageResNetSonarBind',
    num_queries=900,
    with_box_refine=True,
    as_two_stage=True,
    # ── 对比损失配置（与 SonarBind 完全相同） ──
    loss_sonar_image_weight=1.0,
    loss_sonar_text_weight=1.0,
    contrastive_temperature=0.07,
    learnable_temperature=False,
    # ── 数据预处理：使用 ImageNet 归一化参数 ──
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=_imagenet_mean,
        std=_imagenet_std,
        bgr_to_rgb=True,
        pad_mask=False,
        pad_size_divisor=32,   # ResNet stride=32
    ),
    # ── 文本编码器：CLIP ViT-L/14 文本端（hidden_size=768） ──
    language_model=dict(
        type='CLIPModel',
        name=clip_model_path,
        max_tokens=77,          # CLIP 文本最大长度（固定 77）
        pad_to_max=False,
        use_sub_sentence_represent=True,
        special_tokens_list=['<|startoftext|>', '<|endoftext|>', '.', '?'],
    ),
    # ── 图像骨干：ResNet-50（ImageNet 预训练） ──
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),  # C3[512ch], C4[1024ch], C5[2048ch]
        frozen_stages=4,       # 不冻结，全量微调
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/media/fishyu/fish-14tb-2/YiFei/Grounding_DINO/mmdetection/checkpoints/resnet50-0676ba61.pth'),
    ),
    # ── Neck：ResNet 三尺度输入 → 4 尺度 256ch 输出 ──
    # ChannelMapper 处理 ResNet C3/C4/C5，再生成 extra_conv：
    #   convs[0]: [B, 512,  H/8,  W/8]  → [B, 256, H/8,  W/8]   (C3)
    #   convs[1]: [B, 1024, H/16, W/16] → [B, 256, H/16, W/16]  (C4)
    #   convs[2]: [B, 2048, H/32, W/32] → [B, 256, H/32, W/32]  (C5)
    #   extra_conv: [B, 256, H/32, W/32] → [B, 256, H/64, W/64] (P6)
    neck=dict(
        type='ChannelMapper',
        in_channels=[512, 1024, 2048],  # ResNet C3/C4/C5
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        bias=True,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4,
    ),
    # ── 声纳骨干：ResNet-50（独立权重，不与图像分支共享） ──
    sonar_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),      # 只输出 C4：[B, 2048, H/32, W/32]
        frozen_stages=-1,       # 不冻结，全量微调
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/media/fishyu/fish-14tb-2/YiFei/Grounding_DINO/mmdetection/checkpoints/resnet50-0676ba61.pth'),
    ),
    # ── Encoder：完全复用 SonarBind 的 SonarBindTransformerEncoder ──
    encoder=dict(
        type='SonarBindTransformerEncoder',
        num_layers=6,
        num_cp=6,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=4, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0)),
        text_layer_cfg=dict(
            self_attn_cfg=dict(num_heads=4, embed_dims=256, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=1024, ffn_drop=0.0)),
        fusion_layer_cfg=dict(
            v_dim=256,
            l_dim=256,
            embed_dim=1024,
            num_heads=4,
            init_values=1e-4),
    ),
    # ── Decoder：完全复用 SonarBind 的 SonarBindTransformerDecoder ──
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            cross_attn_text_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            cross_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0)),
        post_norm_cfg=None),
    positional_encoding=dict(
        num_feats=128, normalize=True, offset=0.0, temperature=20),
    bbox_head=dict(
        type='GroundingDINOHead',
        num_classes=7,
        sync_cls_avg_factor=True,
        # max_text_len 对齐 CLIP 最大 token 数 77
        contrastive_cfg=dict(max_text_len=77, log_scale='auto', bias=True),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0)),
    dn_cfg=dict(
        label_noise_scale=0.5,
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=100)),
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='BinaryFocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(max_per_img=300))

# ── 数据集 ──────────────────────────────────────────────────────────────────
metainfo = {
    'classes': ('connected_ball', 'connected_polyhedron', 'fake_person',
                'frustum', 'iron_ball', 'octahedron', 'uuv'),
    'palette': [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
                (106, 0, 228), (0, 60, 100), (0, 80, 100)]
}

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadSonarImage', sonar_data_root=sonar_data_root,
         use_zero_fallback=True),
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
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
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
                   'scale_factor', 'flip', 'flip_direction',
                   'sonar_path', 'sonar_shape'))
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None,
         imdecode_backend='pillow'),
    dict(type='LoadSonarImage', sonar_data_root=sonar_data_root,
         use_zero_fallback=True),
    dict(type='FixScaleResize', scale=(800, 1333), keep_ratio=True,
         backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'sonar_path', 'sonar_shape'))
]

dataset_type = 'CocoDataset'

train_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    metainfo=metainfo,
    ann_file='instances_train.json',
    data_prefix=dict(img='train_images/'),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=train_pipeline,
    return_classes=True,
    backend_args=None)

val_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    metainfo=metainfo,
    ann_file='instances_val.json',
    data_prefix=dict(img='val_images/'),
    pipeline=test_pipeline,
    return_classes=True,
    backend_args=None)

train_dataloader = dict(
    _delete_=True,
    batch_size=3,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=train_dataset)

val_dataloader = dict(
    _delete_=True,
    batch_size=3,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=val_dataset)

test_dataloader = val_dataloader

# ── 优化器 ──────────────────────────────────────────────────────────────────
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.0),       # ResNet-50 图像骨干
            'language_model': dict(lr_mult=0.0), # CLIP 文本编码器
            'sonar_backbone': dict(lr_mult=0.01), # ResNet-50 声纳骨干
        }))

# ── 学习率调度 ───────────────────────────────────────────────────────────────
max_epochs = 2  # 先行测试，后续可调整
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[35, 45],
        gamma=0.1)
]

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=2)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ── 评估 ─────────────────────────────────────────────────────────────────────
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'instances_val.json',
    metric='bbox',
    format_only=False,
    backend_args=None)
test_evaluator = val_evaluator

auto_scale_lr = dict(base_batch_size=8, enable=False)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3),
    visualization=dict(type='GroundingVisualizationHook'))

model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=True,
    static_graph=True)
