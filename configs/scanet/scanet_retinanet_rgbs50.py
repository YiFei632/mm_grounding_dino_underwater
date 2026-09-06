"""
SCANet + RetinaNet 在 RGBS50 上的完整配置
用于水下RGB-Sonar双模态目标检测
"""

_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# 自定义导入，确保模块被正确注册
custom_imports = dict(
    imports=['mmdet.models.backbones.scanet_vit',
             'mmdet.datasets.transforms.loading_rgbs'],
    allow_failed_imports=False
)

# ============ 模型配置 ============
model = dict(
    # 注意：4通道数据已在LoadRGBSImageFromFile中归一化
    # 这里禁用预处理器的归一化
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=None,  # 禁用均值归一化
        std=None,   # 禁用标准差归一化
        bgr_to_rgb=False,
        pad_size_divisor=32
    ),
    # 完全替换backbone为SCANetViT（使用_delete_=True清除base配置）
    backbone=dict(
        _delete_=True,  # 删除base中的ResNet配置
        type='SCANetViT',
        img_size=224,
        patch_size=16,
        in_chans=4,          # RGB(3) + Sonar(1)
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        rgbs_loc=[3, 6, 9],  # SCAM融合层位置
        out_indices=(3, 5, 7, 11),  # 输出4个尺度
        norm_eval=False
        # 暂时不加载预训练权重，从头训练
        # init_cfg=dict(
        #     type='Pretrained',
        #     checkpoint='/media/fishyu/fish-14tb-2/YiFei/Grounding_DINO/mmdetection/checkpoints/SCANet_pretrained.pth',
        #     prefix='backbone.'
        # )
    ),
    # FPN配置（完全替换）
    neck=dict(
        _delete_=True,  # 删除base中的配置
        type='FPN',
        in_channels=[768, 768, 768, 768],  # 4个尺度，都是embed_dim=768
        out_channels=256,
        start_level=0,
        add_extra_convs='on_input',
        num_outs=5
    ),
    # RetinaNet Head
    bbox_head=dict(
        num_classes=6  # RGBS50有6个类别
    )
)

# ============ 数据集配置 ============
dataset_type = 'CocoDataset'
data_root = '/media/fishyu/fish-14tb-2/YiFei/Dataset/RGBS50_merged/'

metainfo = {
    'classes': ('connected_polyhedron', 'fake_person', 'frustum', 'iron_ball', 'octahedron', 'uuv'),
    'palette': [(119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228), (0, 60, 100), (0, 80, 100)]
}

backend_args = None

# 训练pipeline
train_pipeline = [
    dict(type='LoadRGBSImageFromFile', to_float32=True, backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),  # 固定224x224
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

# 测试pipeline
test_pipeline = [
    dict(type='LoadRGBSImageFromFile', to_float32=True, backend_args=backend_args),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),  # 固定224x224
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')
    )
]

# Dataloader
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_train.json',
        data_prefix=dict(img='train/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_val.json',
        data_prefix=dict(img='val/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args
    )
)

test_dataloader = val_dataloader

# ============ 评估配置 ============
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val.json',
    metric='bbox',
    backend_args=backend_args
)

test_evaluator = val_evaluator

# ============ 训练配置 ============
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=20, val_interval=1)

# 优化器（完全替换base配置）
optim_wrapper = dict(
    _delete_=True,  # 删除base中的SGD配置
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2)
)

# 学习率策略
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=500
    ),
    dict(
        type='MultiStepLR',
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1,
        begin=0,
        end=12
    )
]

# ============ 其他配置 ============

# ============ 其他配置 ============
work_dir = './work_dirs/scanet_retinanet_rgbs50'

# 日志
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'

# 加载和恢复
load_from = None
resume = False

# 随机种子
randomness = dict(seed=42, deterministic=False)

