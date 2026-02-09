_base_ = [                                                                                                                                                                                                          
    '../_base_/datasets/utdac2020_detection.py',  # 需要创建DUO数据集配置                                                                                                                                                                                                                                                                                                                           
    '../_base_/default_runtime.py'                                                                                                                                                                                  
]                                                                                                                                                                                                                   
data_root = '/media/fishyu/6955024a-ed66-4a86-b94a-687c51c28306/fishyu/YiFei/Datasets/UTDAC2020/'
class_name = ('echinus', 'starfish', 'holothurian', 'scallop')
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230)])

num_levels = 5                                                                                                                                                                                                                    
# 模型配置                                                                                                                                                                                                          
model = dict(                                                                                                                                                                                                       
    type='UCFDINO',                                                                                                                                                                                                 
    num_queries=900,                                                                                                                                                                                                
    with_box_refine=True,                                                                                                                                                                                           
    as_two_stage=True,                                                                                                                                                                                              
    data_preprocessor=dict(                                                                                                                                                                                         
        type='DetDataPreprocessor',                                                                                                                                                                                 
        mean=[123.675, 116.28, 103.53],                                                                                                                                                                             
        std=[58.395, 57.12, 57.375],                                                                                                                                                                                
        bgr_to_rgb=True,                                                                                                                                                                                            
        pad_size_divisor=1),                                                                                                                                                                                        
                                                                                                                                                                                                                    
    # 图像增强器配置                                                                                                                                                                                                
    enhancer=dict(                                                                                                                                                                                                  
        type='UnderwaterImageEnhancer',                                                                                                                                                                             
        method='color_correction',                                                                                                                                                                                  
        pretrained=None  # 可选：预训练增强模型路径                                                                                                                                                                 
    ),                                                                                                                                                                                                              
                                                                                                                                                                                                                    
    # 是否在编码器中融合双流                                                                                                                                                                                        
    fusion_in_encoder=False,  # 禁用以节省内存,                                                                                                                                                                                         
                                                                                                                                                                                                                    
    # Backbone                                                                                                                                                                                                      
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
                                                                                                                                                                                                                    
    # Neck                                                                                                                                                                                                          
    neck=dict(                                                                                                                                                                                                      
        type='ChannelMapper',                                                                                                                                                                                       
        in_channels=[512, 1024, 2048],                                                                                                                                                                              
        kernel_size=1,                                                                                                                                                                                              
        out_channels=256,                                                                                                                                                                                           
        act_cfg=None,                                                                                                                                                                                               
        norm_cfg=dict(type='GN', num_groups=32),                                                                                                                                                                    
        num_outs=4),                                                                                                                                                                                                
                                                                                                                                                                                                                    
    # Encoder                                                                                                                                                                                                       
    encoder=dict(                                                                                                                                                                                                   
        num_layers=6,                                                                                                                                                                                               
        layer_cfg=dict(                                                                                                                                                                                             
            self_attn_cfg=dict(                                                                                                                                                                                     
                embed_dims=256,                                                                                                                                                                                     
                num_heads=8,                                                                                                                                                                                        
                dropout=0.0,                                                                                                                                                                                        
                batch_first=True),                                                                                                                                                                                  
            ffn_cfg=dict(                                                                                                                                                                                           
                embed_dims=256,                                                                                                                                                                                     
                feedforward_channels=2048,                                                                                                                                                                          
                num_fcs=2,                                                                                                                                                                                          
                ffn_drop=0.0,                                                                                                                                                                                       
                act_cfg=dict(type='ReLU', inplace=True)))),                                                                                                                                                         
                                                                                                                                                                                                                    
    # Decoder                                                                                                                                                                                                       
    decoder=dict(                                                                                                                                                                                                   
        num_layers=6,                                                                                                                                                                                               
        return_intermediate=True,                                                                                                                                                                                   
        layer_cfg=dict(                                                                                                                                                                                             
            self_attn_cfg=dict(                                                                                                                                                                                     
                embed_dims=256,                                                                                                                                                                                     
                num_heads=8,                                                                                                                                                                                        
                dropout=0.0,                                                                                                                                                                                        
                batch_first=True),                                                                                                                                                                                  
            cross_attn_cfg=dict(                                                                                                                                                                                    
                embed_dims=256,                                                                                                                                                                                     
                num_heads=8,                                                                                                                                                                                        
                dropout=0.0,                                                                                                                                                                                        
                batch_first=True),                                                                                                                                                                                  
            ffn_cfg=dict(                                                                                                                                                                                           
                embed_dims=256,                                                                                                                                                                                     
                feedforward_channels=2048,                                                                                                                                                                          
                num_fcs=2,                                                                                                                                                                                          
                ffn_drop=0.0,                                                                                                                                                                                       
                act_cfg=dict(type='ReLU', inplace=True))),                                                                                                                                                          
        post_norm_cfg=None),                                                                                                                                                                                        
                                                                                                                                                                                                                    
    # 位置编码                                                                                                                                                                                                      
    positional_encoding=dict(                                                                                                                                                                                       
        num_feats=128,                                                                                                                                                                                              
        normalize=True,                                                                                                                                                                                             
        offset=0.0,                                                                                                                                                                                                 
        temperature=20),                                                                                                                                                                                            
                                                                                                                                                                                                                    
    # BBox Head                                                                                                                                                                                                     
    bbox_head=dict(                                                                                                                                                                                                 
        type='DINOHead',                                                                                                                                                                                            
        num_classes=4,  # 根据数据集修改                                                                                                                                                                           
        sync_cls_avg_factor=True,                                                                                                                                                                                   
        loss_cls=dict(                                                                                                                                                                                              
            type='FocalLoss',                                                                                                                                                                                       
            use_sigmoid=True,                                                                                                                                                                                       
            gamma=2.0,                                                                                                                                                                                              
            alpha=0.25,                                                                                                                                                                                             
            loss_weight=1.0),                                                                                                                                                                                       
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),                                                                                                                                                             
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),                                                                                                                                                           
                                                                                                                                                                                                                    
    # DN配置                                                                                                                                                                                                        
    dn_cfg=dict(                                                                                                                                                                                                    
        label_noise_scale=0.5,                                                                                                                                                                                      
        box_noise_scale=1.0,                                                                                                                                                                                        
        group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=100)),                                                                                                                                         
                                                                                                                                                                                                                    
    # 训练和测试配置                                                                                                                                                                                                
    train_cfg=dict(                                                                                                                                                                                                 
        assigner=dict(                                                                                                                                                                                              
            type='HungarianAssigner',                                                                                                                                                                               
            match_costs=[                                                                                                                                                                                           
                dict(type='FocalLossCost', weight=2.0),                                                                                                                                                             
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),                                                                                                                                             
                dict(type='IoUCost', iou_mode='giou', weight=2.0)                                                                                                                                                   
            ])),                                                                                                                                                                                                    
    test_cfg=dict(max_per_img=300))                                                                                                                                                                                 
                                                                                                                                                                                                                    
# 优化器                                                                                                                                                                                                            
optim_wrapper = dict(                                                                                                                                                                                               
    type='OptimWrapper',                                                                                                                                                                                            
    optimizer=dict(                                                                                                                                                                                                 
        type='AdamW',                                                                                                                                                                                               
        lr=0.00005,  # 减半学习率因为batch size减半 (原来0.0001)                                                                                                                                                                                                  
        weight_decay=0.0001),                                                                                                                                                                                       
    clip_grad=dict(max_norm=0.1, norm_type=2),                                                                                                                                                                      
    paramwise_cfg=dict(                                                                                                                                                                                             
        custom_keys={                                                                                                                                                                                               
            'backbone': dict(lr_mult=0.1),                                                                                                                                                                          
            'enhancer': dict(lr_mult=0.0),  # 冻结增强器                                                                                                                                                            
        }))                                                                                                                                                                                                         
                                                                                                                                                                                                                    
# 学习率调度                                                                                                                                                                                                        
param_scheduler = [                                                                                                                                                                                                 
    dict(                                                                                                                                                                                                           
        type='MultiStepLR',                                                                                                                                                                                         
        begin=0,                                                                                                                                                                                                    
        end=12,                                                                                                                                                                                                     
        by_epoch=True,                                                                                                                                                                                              
        milestones=[11],                                                                                                                                                                                            
        gamma=0.1)                                                                                                                                                                                                  
]                                                                                                                                                                                                                   

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(320, 800), (352, 800), (384, 800), (416, 800),
                            (448, 800), (480, 800)],  # 减小分辨率以节省内存
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(400, 2000), (500, 2000), (600, 2000)],  # 减小分辨率,
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(320, 480),  # 减小crop size,
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(320, 800), (352, 800), (384, 800), (416, 800),
                            (448, 800), (480, 800)],
                    keep_ratio=True)
            ]
        ]),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities'))
]

train_dataloader = dict(
    batch_size=1,  # 减小batch size从2到1以节省内存
    dataset=dict(
        _delete_=True,
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        return_classes=True,
        pipeline=train_pipeline,
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/')))

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/')))

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'annotations/instances_val2017.json')
test_evaluator = val_evaluator

max_epoch = 24

# 训练配置                                                                                                                                                                                                          
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=1)                                                                                                                                         
val_cfg = dict(type='ValLoop')                                                                                                                                                                                      
test_cfg = dict(type='TestLoop')                                                                                                                                                                                    

load_from = '/media/fishyu/6955024a-ed66-4a86-b94a-687c51c28306/fishyu/YiFei/Grounding_DINO/mmdetection/checkpoints/ucf_dino_vision_only.pth'                                                                                                                                                                                                    


# DDP配置 - 处理未使用的参数（如冻结的enhancer）
find_unused_parameters = True


# DDP wrapper配置
model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=True,
    static_graph=False  # 不使用静态图，因为模型结构可能变化
)
