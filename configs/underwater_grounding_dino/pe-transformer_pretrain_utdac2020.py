_base_ = [
    '../_base_/datasets/utdac2020_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# 1. 注册你的自定义模块 (假设你已经按之前步骤放好了文件)
custom_imports = dict(
    imports=['mmdet.models.backbones.cswin', 
             'mmdet.models.necks.pe_transformer_neck',
             'mmdet.models.dense_heads.pe_transformer_head'],
    allow_failed_imports=False)

model = dict(
    type='RepPointsDetector', # 使用支持 Point 的检测器架构
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
        
    # ================= Backbone =================
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

    # ================= Neck =================
    neck=dict(
        type='PETransformerNeck',
        in_channels=[64, 128, 256, 512],
        out_channels=256
    ),

    # ================= Decoder (Head) =================
    # 使用刚才写的带 ASPP 的 RepPointsHead
    bbox_head=dict(
        type='PETransformerHead', 
        num_classes=80, # 改为你的水下数据集类别数
        in_channels=256,
        feat_channels=256,
        point_feat_channels=256,
        stacked_convs=3,
        num_points=9,  # 自适应点集数量
        gradient_mul=0.1,
        point_strides=[4, 8, 16, 32], # 对应 Neck 的 P3, P4, P5 (RepPoints 通常用3层)
        # 如果你的 Neck 输出了 4 层 (4,8,16,32)，你需要在这里匹配 strides
        
        # --- Loss Function (对应论文 Eq 6) ---
        loss_cls=dict(
            type='FocalLoss', # 论文 cite: 269 使用 Focal Loss
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox_init=dict(type='GIoULoss', loss_weight=0.375),
        loss_bbox_refine=dict(type='GIoULoss', loss_weight=1.0),
        # 空间约束由 RepPoints 的 PointGenerator 和 Assign 隐式处理
        transform_method='moment', # 对应论文 cite: 260 "conversion function"
        # momentum=0.9, # 用于 moment 转换
        # use_snap_loss=True, # 对应论文中的 spatial constraint 思想
        # loss_snap=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0) # 空间对齐损失
    ),
    
    # ================= ASAA (Assigner) =================
    # 对应论文 3.4.2 Dynamic Assessment and Assignment
    train_cfg=dict(
        init=dict(
            assigner=dict(type='PointAssigner', scale=4, pos_num=1),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        refine=dict(
            # 这里对应 "Dynamic K" 和 "Iterative"
            # RepPoints 使用 MaxIoUAssigner 进行第二阶段精细分配
            assigner=dict(
                type='MaxIoUAssigner', 
                pos_iou_thr=0.5,
                neg_iou_thr=0.4,
                min_pos_iou=0,
                ignore_iof_thr=-1),
            allowed_border=-1,
            pos_weight=-1,
            debug=False)
    ),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100)
)

# 优化器配置 (AdamW)
optim_wrapper = dict(
    _delete_=True,
    optimizer=dict(type='AdamW', lr=0.0010, weight_decay=0.05)
)