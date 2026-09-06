_base_ = './mask-rcnn_x101-32x4d_fpn_2x_coco.py'
model = dict(
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint='/media/fishyu/fish-14tb-2/YiFei/Grounding_DINO/mmdetection/checkpoints/resnext101_64x4d-ee2c6f71.pth')))
