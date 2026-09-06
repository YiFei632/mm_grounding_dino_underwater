"""
RGBS50数据集配置
"""

# 数据集设置
dataset_type = 'CocoDataset'
data_root = '/media/fishyu/fish-14tb-2/YiFei/Dataset/RGBS50_merged/'

# 类别设置（单类别：underwater object）
metainfo = {
    'classes': ('object',),
    'palette': [(220, 20, 60)]
}

# 训练数据增强pipeline
train_pipeline = [
    dict(type='LoadRGBSImageFromFile', to_float32=True),  # 自定义：加载4通道图像
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        scale=(640, 640),
        keep_ratio=True
    ),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

# 测试pipeline
test_pipeline = [
    dict(type='LoadRGBSImageFromFile', to_float32=True),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')
    )
]

# 训练数据加载器
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
        pipeline=train_pipeline
    )
)

# 验证数据加载器
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
        pipeline=test_pipeline
    )
)

# 测试数据加载器
test_dataloader = val_dataloader

# 评估器
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val.json',
    metric='bbox',
    format_only=False
)

test_evaluator = val_evaluator
