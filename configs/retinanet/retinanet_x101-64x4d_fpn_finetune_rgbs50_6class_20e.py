_base_ = ['../_base_/models/retinanet_r50_fpn.py', '../common/ms_rgbs50.py']
# optimizer
model = dict(
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        init_cfg=dict(
            type='Pretrained', checkpoint='/media/fishyu/fish-14tb-2/YiFei/Grounding_DINO/mmdetection/checkpoints/resnext101_64x4d-ee2c6f71.pth')))
optim_wrapper = dict(optimizer=dict(type='SGD', lr=0.01))

load_from = '/media/fishyu/fish-14tb-2/YiFei/Grounding_DINO/mmdetection/checkpoints/retinanet_x101_64x4d_fpn_mstrain_3x_coco_20210719_051838-022c2187.pth'