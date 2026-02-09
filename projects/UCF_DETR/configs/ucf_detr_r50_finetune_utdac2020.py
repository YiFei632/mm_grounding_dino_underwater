_base_ = [
    'mmdet::_base_/datasets/utdac2020_detection.py',
    'mmdet::_base_/default_runtime.py',
    'mmdet::_base_/schedules/schedule_1x.py',
]

custom_imports = dict(
    imports=['projects.UCF_DETR.ucf_detector', 'projects.UCF_DETR.ucf_neck', 
             'projects.UCF_DETR.ucf_enhancer', 'projects.UCF_DETR.ucf_decoder'],
    allow_failed_imports=False)

load_from = '/media/fishyu/6955024a-ed66-4a86-b94a-687c51c28306/fishyu/YiFei/Grounding_DINO/mmdetection/checkpoints/UCF_DETR_Pretrain_UDD_mmdet.pth'

model = dict(
    type='UCFRTDETR',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    
    enhancer_cfg=dict(sigma=[30, 150, 300], restore_factor=2.0),

    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='/media/fishyu/6955024a-ed66-4a86-b94a-687c51c28306/fishyu/YiFei/Grounding_DINO/mmdetection/checkpoints/resnet50-0676ba61.pth')),

    neck=dict(
        type='UCFHybridEncoder',
        in_channels=[512, 1024, 2048],
        hidden_dim=256,
        feat_strides=[8, 16, 32]),

    bbox_head=dict(
        type='UCFRTDETRHead',
        num_classes=4,
        in_channels=256,
        transformer=dict(
            type='UCFTransformerDecoder', # MMDet 原生
            return_intermediate=True,
            num_layers=6,
            layer_cfg=dict(
                type='UCFDetrTransformerDecoderLayer', # 自定义 CQI Layer
                d_model=256,
                n_head=8,
                dropout=0.0,
                # MMDet 需要的 attn config
                self_attn_cfg=dict(embed_dims=256, num_heads=8),
                cross_attn_cfg=dict(embed_dims=256, num_levels=3),
                ffn_cfg=dict(embed_dims=256, feedforward_channels=1024),
            )
        ),
        # 这里的 Loss 设置参考 RT-DETR
        loss_cls=dict(type='QualityFocalLoss', use_sigmoid=True, beta=2.0, loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)
    ),
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(max_per_img=300)
)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0005, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1)}
    )
)

model_wrapper_cfg = dict(type='MMDistributedDataParallel', find_unused_parameters=True)