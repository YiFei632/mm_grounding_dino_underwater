_base_ = '../../dino/dino-4scale_r50_8xb2-12e_coco.py'

data_root = '/home/user/YiFei/Datasets/RGBS50_image/'
class_name = ('ball_and_polyhedron', 'connected_polyhedron', 'fake_person', 'frustum', 'iron_ball', 'octahedron', 'uuv')
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (119, 11, 32), (0, 0, 142), (0, 0, 230)])

pretrained = '/home/user/YiFei/Grounding_DINO/mm_grounding_dino_underwater/checkpoints/swin_large_patch4_window12_384_22k.pth'  # noqa
num_levels = 5
model = dict(
    num_feature_levels=num_levels,
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        # Please only add indices that would be used
        # in FPN, otherwise some parameter will not be used
        with_cp=True,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[192, 384, 768, 1536], num_outs=num_levels),
    encoder=dict(layer_cfg=dict(self_attn_cfg=dict(num_levels=num_levels))),
    decoder=dict(layer_cfg=dict(cross_attn_cfg=dict(num_levels=num_levels))),
    bbox_head=dict(num_classes=7))

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
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
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
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities'))
]

train_dataloader = dict(
    batch_size=1,  # 每卡1个样本，4卡总共4样本
    num_workers=2,
    persistent_workers=True,
    dataset=dict(
        _delete_=True,
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        return_classes=True,
        pipeline=train_pipeline,
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        ann_file='instances_train.json',
        data_prefix=dict(img='train_images/')))

val_dataloader = dict(
    batch_size=1,  # 每卡1个样本，4卡总共4样本
    num_workers=2,
    persistent_workers=True,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='instances_val.json',
        data_prefix=dict(img='val_images/')))

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'instances_val.json')
test_evaluator = val_evaluator

max_epoch = 2

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=1, save_best='auto'),
    logger=dict(type='LoggerHook', interval=5))
train_cfg = dict(max_epochs=max_epoch, val_interval=1)

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epoch,
        by_epoch=True,
        milestones=[100],
        gamma=0.1)
]

optim_wrapper = dict(
    type='OptimWrapper',  # 改用普通优化器，避免AMP数值不稳定
    optimizer=dict(
        type='AdamW',
        lr=0.0001,  # 降低学习率从0.0005到0.0001，避免数值爆炸
        weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)}))