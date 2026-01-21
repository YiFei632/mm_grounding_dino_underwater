# =========================================================
# 1. 基础配置引入 (根据你的环境选择合适的基础配置)
# =========================================================
_base_ = [
    '../_base_/datasets/rgbs50_detection.py',       # 替换为你自己的水下数据集配置
    '../_base_/schedules/schedule_1x.py',         # 1x 训练策略 (12 epochs)
    '../_base_/default_runtime.py'
]

# =========================================================
# 2. 模型结构配置 (PE-Transformer + RetinaNet Head)
# =========================================================
model = dict(
    type='RetinaNet',  # 使用 RetinaNet 架构
    
    # --- 数据预处理 ---
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=False,  # RetinaNet 不需要 mask
        pad_size_divisor=32),

    # --- Backbone: CSWin-Transformer ---
    #
    backbone=dict(
        type='CSWinTransformer',  # 确保这与你注册的类名一致
        embed_dim=64,
        depth=[1, 2, 21, 1],
        num_heads=[2, 4, 8, 16],
        split_size=[1, 2, 7, 7],
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        use_chk=False,  # 显存不够可设为 True (checkpointing)
        
        # 【关键】加载你下载的 ckpt
        init_cfg=dict(
            type='Pretrained', 
            checkpoint='/media/fishyu/6955024a-ed66-4a86-b94a-687c51c28306/fishyu/YiFei/Grounding_DINO/mmdetection/checkpoints/cswin_base_pretrain_imagenet_224.pth' # 修改为你下载的文件路径
        )
    ),

    # --- Neck: PE-Transformer Neck ---
    neck=dict(
        type='PETransformerNeck', # 你之前定义的类名
        in_channels=[64, 128, 256, 512],
        out_channels=256
    ),

    # --- Decoder/Head: RetinaNet Head ---
    bbox_head=dict(
        type='RetinaHead',
        num_classes=7,  # 【修改】改为你的水下数据集类别数
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        
        # 【关键适配】你的 Neck 输出 Stride 为 [4, 8, 16, 32]
        # 标准 RetinaNet 通常是 [8, 16, 32, 64, 128]，所以这里必须显式指定
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32]  # 必须与 Neck 输出对齐
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        
    # --- 训练和测试配置 ---
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100)
)

# =========================================================
# 3. 优化器配置 (参考 CSWin 官方推荐)
# =========================================================
optim_wrapper = dict(
    _delete_=True,
    optimizer=dict(
        type='AdamW',
        lr=0.0001,       # 初始学习率，根据 Batch Size 调整
        weight_decay=0.05,
        betas=(0.9, 0.999),
    ),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }
    ),
    # 梯度裁剪，防止训练初期不稳定
    clip_grad=dict(max_norm=20.0, norm_type=2)
)