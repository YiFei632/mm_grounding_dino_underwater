_base_ = [
    '../../_base_/datasets/coco_detection.py',
    '../../_base_/schedules/schedule_1x.py', '../../_base_/default_runtime.py'
]

# Custom imports to ensure all transforms are registered
custom_imports = dict(
    imports=['mmdet.datasets', 'mmdet.models'],
    allow_failed_imports=False)

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa
lang_model_name = '/media/fishyu/fish-14tb-2/YiFei/Grounding_DINO/mmdetection/clip-vit-base-patch32'

model = dict(
    type='ThirdGroundingDINO',
    num_queries=900,
    with_box_refine=True,
    as_two_stage=True,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=False,
    ),
    language_model=dict(                                                                                                                                                                                         
        type='CLIPModel',  # 新的类型                                                                                                                                                                        
        name=lang_model_name,  # 或其他CLIP模型                                                                                                                                                   
        max_tokens=77,  # CLIP的最大token长度是77                                                                                                                                                                
        pad_to_max=False,                                                                                                                                                                                        
        use_sub_sentence_represent=True,                                                                                                                                                                         
        special_tokens_list=['<|startoftext|>', '<|endoftext|>', '.', '?'],                                                                                                                                      
    ),
    backbone=dict(
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),
        with_cp=True,
        convert_weights=True,
        frozen_stages=-1,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    sonar_backbone=dict(                                                                                                                                                                                                             
        type='ResNet',                                                                                                                                                                                                               
        depth=50,                                                                                                                                                                                                                    
        num_stages=4,                                                                                                                                                                                                                
        out_indices=(3,),  # 只要最后一层特征                                                                                                                                                                                        
        frozen_stages=1,                                                                                                                                                                                                             
        norm_cfg=dict(type='BN', requires_grad=True),                                                                                                                                                                                
        norm_eval=True,                                                                                                                                                                                                              
        style='pytorch',                                                                                                                                                                                                             
        init_cfg=dict(type='Pretrained', checkpoint='/media/fishyu/fish-14tb-2/YiFei/Grounding_DINO/mmdetection/checkpoints/resnet50-0676ba61.pth')                                                                                                                                                        
    ),
    neck=dict(
        type='ChannelMapper',
        in_channels=[192, 384, 768],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        bias=True,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    encoder=dict(
        type='ThirdGroundingDinoTransformerEncoder',
        num_layers=6,
        num_cp=6,
        # visual layer config
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=4, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0)),
        # text layer config
        text_layer_cfg=dict(
            self_attn_cfg=dict(num_heads=4, embed_dims=256, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=1024, ffn_drop=0.0)),
        # fusion layer config
        fusion_layer_cfg=dict(
            v_dim=256,
            l_dim=256,
            embed_dim=1024,
            num_heads=4,
            init_values=1e-4),
    ),
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            # query self attention layer
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            # cross attention layer query to text
            cross_attn_text_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            # cross attention layer query to image
            cross_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0)),
        post_norm_cfg=None),
    positional_encoding=dict(
        num_feats=128, normalize=True, offset=0.0, temperature=20),
    bbox_head=dict(
        type='GroundingDINOHead',
        num_classes=7,  # RGBS50有7个类别
        sync_cls_avg_factor=True,
        contrastive_cfg=dict(max_text_len=256, log_scale='auto', bias=True),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0)),
    dn_cfg=dict(  # TODO: Move to model.train_cfg ?
        label_noise_scale=0.5,
        box_noise_scale=1.0,  # 0.4 for DN-DETR
        group_cfg=dict(dynamic=True, num_groups=None,
                       num_dn_queries=100)),  # TODO: half num_dn_queries
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='BinaryFocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(max_per_img=300))

# dataset settings
# 声纳数据集根目录
sonar_data_root = '/media/fishyu/fish-14tb-2/YiFei/Dataset/RGBS50_sonar'

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadSonarImage', sonar_data_root=sonar_data_root, use_zero_fallback=True),
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
    dict(type='LoadImageFromFile', backend_args=None, imdecode_backend='pillow'),
    dict(type='LoadSonarImage', sonar_data_root=sonar_data_root, use_zero_fallback=True),
    dict(type='FixScaleResize', scale=(800, 1333), keep_ratio=True, backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'sonar_path', 'sonar_shape'))
]

dataset_type = 'CocoDataset'
data_root = '/media/fishyu/fish-14tb-2/YiFei/Dataset/RGBS50_image/'

# RGBS50数据集类别定义
metainfo = {
    'classes': ('ball_and_polyhedron', 'connected_polyhedron', 'fake_person', 'frustum',
                'iron_ball', 'octahedron', 'uuv'),
    'palette': [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
                (106, 0, 228), (0, 60, 100), (0, 80, 100)]
}

# 训练数据集
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

# 验证数据集
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
    batch_size=2,  # 双图像输入，减小batch_size
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

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001),  # 降低学习率，因为batch_size更小
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.1),
            'language_model': dict(lr_mult=0.1),
            'sonar_backbone': dict(lr_mult=0.1),
        }))

# learning policy
max_epochs = 2  # 先行进行测试，训练两轮之后微调看效果
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[35, 45],  # 调整milestone
        gamma=0.1)
]

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=5)  # 每5个epoch验证一次

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 评估指标
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'instances_val.json',
    metric='bbox',
    format_only=False,
    backend_args=None)

test_evaluator = val_evaluator

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (4 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=8, enable=False)  # 关闭自动学习率缩放

default_hooks = dict(visualization=dict(type='GroundingVisualizationHook'))

model_wrapper_cfg = dict(                                                                                                                                                                                    
    type='MMDistributedDataParallel',                                                                                                                                                                        
    find_unused_parameters=True,
    static_graph=True                                                                                                                                                                           
)
