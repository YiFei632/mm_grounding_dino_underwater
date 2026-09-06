"""
训练配置 - 12 epochs
适用于在预训练模型基础上微调
"""

# 优化器
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
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
        end=500  # warmup 500 iterations
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

# 训练设置
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=12,
    val_interval=1
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
