"""
SCANet模型配置 - 基础配置
"""

_base_ = [
    '../../../_base_/models/retinanet_r50_fpn.py',
]

# 模型配置
model = dict(
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53, 127.5],  # RGB + Sonar均值
        std=[58.395, 57.12, 57.375, 30.0],      # RGB + Sonar标准差
        bgr_to_rgb=True,
        pad_size_divisor=32
    ),

    # 替换backbone为SCANetViT
    backbone=dict(
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
        norm_eval=False,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/media/fishyu/fish-14tb-2/YiFei/Grounding_DINO/mmdetection/checkpoints/SCANet_pretrained.pth',
            prefix='backbone.'
        )
    ),

    # Neck: FPN
    neck=dict(
        type='FPN',
        in_channels=[768, 768, 768, 768],  # 4个尺度，都是embed_dim=768
        out_channels=256,
        start_level=0,
        add_extra_convs='on_input',
        num_outs=5
    ),

    # Detection Head: RetinaNet Head
    bbox_head=dict(
        type='RetinaHead',
        num_classes=1,  # RGBS50是单类别
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]
        ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0
        ),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)
    ),

    # Training and testing cfg
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1
        ),
        sampler=dict(type='PseudoSampler'),
        allowed_border=-1,
        pos_weight=-1,
        debug=False
    ),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100
    )
)
