"""
SCANet + RetinaNet 在 RGBS50 上的配置
使用ImageNet预训练的ViT-Base扩展到4通道
"""

_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# 自定义导入
custom_imports = dict(
    imports=['mmdet.models.backbones.scanet_vit',
             'mmdet.datasets.transforms.loading_rgbs'],
    allow_failed_imports=False
)

# ============ 模型配置 ============
model = dict(
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=None,  # 数据已在加载时归一化
        std=None,
        bgr_to_rgb=False,
        pad_size_divisor=32
    ),

    backbone=dict(
        _delete_=True,
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
        rgbs_loc=[3, 6, 9],
        out_indices=(3, 5, 7, 11),
        norm_eval=False,
        # ✨ 使用ImageNet预训练的ViT（扩展到4通道）
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/media/fishyu/fish-14tb-2/YiFei/Grounding_DINO/mmdetection/checkpoints/vit_base_4channel.pth',  # 你需要填入实际路径
            prefix=''
        )
    ),

    neck=dict(
        _delete_=True,
        type='FPN',
        in_channels=[768, 768, 768, 768],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_input',
        num_outs=5
    ),

    bbox_head=dict(
        num_classes=4
    )
)

# ============ 数据集配置 ============
dataset_type = 'CocoDataset'
data_root = '/media/fishyu/fish-14tb-2/YiFei/Datasets/RGBS20_merged/'

metainfo = {
    'classes': ('fish', 'fishing_net', 'mine',
                'stick'),
    'palette': [(119, 11, 32), (0, 0, 142), (0, 0, 230),
                (106, 0, 228)]
}

backend_args = None

train_pipeline = [
    dict(type='LoadRGBSImageFromFile', to_float32=True, backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadRGBSImageFromFile', to_float32=True, backend_args=backend_args),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')
    )
]

train_dataloader = dict(
    batch_size=16,  # 使用预训练后可以增加batch size
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

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val.json',
    metric='bbox',
    backend_args=backend_args
)

test_evaluator = val_evaluator

# ============ 训练配置 ============
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=30, val_interval=1)

# 使用预训练权重后，降低学习率
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.00005,  # 降低学习率（从0.0001降到0.00005）
        weight_decay=0.0001
    ),
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
        milestones=[20, 27],  # 在20和27轮时降低学习率
        gamma=0.1,
        begin=0,
        end=30
    )
]

# ============ 其他配置 ============
work_dir = './work_dirs/scanet_retinanet_finetune_12e_rgbs20'

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'

load_from = None
resume = False

randomness = dict(seed=42, deterministic=False)
