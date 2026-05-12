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

# ── CLIP 归一化参数（× 255，RGB 顺序）──────────────────────────────────────
# CLIP 原始（RGB）：mean=[0.48145466, 0.4578275, 0.40821073]
#                   std =[0.26862954, 0.26130258, 0.27577711]
# data_preprocessor 先 bgr_to_rgb=True，再用下方 RGB 顺序的 mean/std
_clip_mean = [122.77, 116.75, 104.09]
_clip_std = [68.50, 66.63, 70.32]

model = dict(
    type='CLIPSonarBind',
    num_queries=900,
    with_box_refine=True,
    as_two_stage=True,
    # ── 对比损失配置（与 SonarBind 完全相同） ──
    loss_sonar_image_weight=1.0,
    loss_sonar_text_weight=1.0,
    contrastive_temperature=0.07,
    learnable_temperature=False,
    # ── 数据预处理：使用 CLIP 归一化参数 ──
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=_clip_mean,
        std=_clip_std,
        bgr_to_rgb=True,
        pad_mask=False,
        pad_size_divisor=14,   # ViT patch_size=14，替换原来的 32
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
    # ── 图像骨干：CLIP ViT-L/14 视觉端（独立权重 A） ──
    backbone=dict(
        type='CLIPViTBackbone',
        model_name=clip_model_path,
        img_size=448,          # 448 / 14 = 32 → 32×32 = 1024 patch tokens
        frozen_stages=0,       # 不冻结，全量微调；如需冻结可设为 24
        use_checkpoint=True,   # 开启 gradient checkpointing 节省显存
        init_cfg=None,         # 权重由 model_name 的 from_pretrained 自动加载
    ),
    # ── Neck：单尺度 1024ch 输入 → 4 尺度 256ch 输出 ──
    # ChannelMapper 自动生成 extra_conv（stride=2 级联）：
    #   convs[0]:    [B, 1024, 32, 32] → [B, 256, 32, 32]   (N=1024)
    #   extra_conv0: [B, 1024, 32, 32] → [B, 256, 16, 16]   (N=256)
    #   extra_conv1: [B,  256, 16, 16] → [B, 256,  8,  8]   (N=64)
    #   extra_conv2: [B,  256,  8,  8] → [B, 256,  4,  4]   (N=16)
    #   N_img_total = 1024 + 256 + 64 + 16 = 1360
    neck=dict(
        type='ChannelMapper',
        in_channels=[1024],    # CLIPViTBackbone 输出单尺度 1024ch
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        bias=True,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4,
    ),
    # ── 声纳骨干：CLIP ViT-L/14 视觉端（独立权重 B，不与图像分支共享） ──
    sonar_backbone=dict(
        type='CLIPViTBackbone',
        model_name=clip_model_path,
        img_size=448,
        frozen_stages=0,
        use_checkpoint=True,
        init_cfg=None,
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
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=train_dataset)

val_dataloader = dict(
    _delete_=True,
    batch_size=1,
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
            'backbone': dict(lr_mult=0.1),       # CLIP 图像骨干
            'language_model': dict(lr_mult=0.1), # CLIP 文本编码器
            'sonar_backbone': dict(lr_mult=0.1), # CLIP 声纳骨干
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
