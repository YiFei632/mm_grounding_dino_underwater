# Preserve the single-fusion model, losses, datasets and batch sizes exactly.
_base_ = ['./oag_double_resnet50_single_fusion_rgbs50_12e.py']

model = dict(bbox_head=dict(num_classes=4))
work_dir = './work_dirs/oag_train/single_fusion_rgbs20_finetune_12e'
load_from = '/media/fishyu/fish-14tb-2/YiFei/Grounding_DINO/mmdetection/work_dirs/oag_train/single_fusion_rgbs50/epoch_12.pth'
# Start a fresh 20-epoch optimization schedule from the trained weights.
resume = False

max_epochs = 12
train_cfg = dict(max_epochs=max_epochs, val_interval=1)
optim_wrapper = dict(optimizer=dict(lr=5e-5))
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', by_epoch=True, begin=0, end=max_epochs,
         milestones=[8, 11], gamma=0.1),
]

dataset_type = 'CocoDataset'
data_root = '/media/fishyu/fish-14tb-2/YiFei/Datasets/RGBS20-RGB/'
sonar_data_root = '/media/fishyu/fish-14tb-2/YiFei/Datasets/RGBS20-Sonar/'
sonar_ann_file = sonar_data_root + 'SONAR_train.json'
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='LoadSonarAnnotations', ann_file=sonar_data_root + 'SONAR_train.json'),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoiceResize',
        scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                (736, 1333), (768, 1333), (800, 1333)],
        keep_ratio=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='LoadSonarImage', sonar_data_root=sonar_data_root + 'Images/',
         flat_layout=True,
         color_type='grayscale', use_zero_fallback=False),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction',
                   'sonar_path', 'sonar_shape'))
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None, imdecode_backend='pillow'),
    dict(type='FixScaleResize', scale=(800, 1333), keep_ratio=True, backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='LoadSonarImage', sonar_data_root=sonar_data_root + 'Images/',
         flat_layout=True,
         color_type='grayscale', use_zero_fallback=False),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'sonar_path', 'sonar_shape'))
]

# RGBS50数据集类别定义
metainfo = {
    'classes': ('fish', 'fishing_net', 'mine',
                'stick'),
    'palette': [(119, 11, 32), (0, 0, 142), (0, 0, 230),
                (106, 0, 228)]
}

# 训练数据集
train_dataset = dict(
    _delete_=True,
    type=dataset_type,
    data_root=data_root,
    metainfo=metainfo,
    ann_file='/media/fishyu/fish-14tb-2/YiFei/Grounding_DINO/mmdetection/configs/underwater_grounding_dino/oag_train/annotations/rgbs20_train_paired.json',
    data_prefix=dict(img='Images/'),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=train_pipeline,
    return_classes=True,
    backend_args=None)

# 验证数据集
val_dataset = dict(
    _delete_=True,
    type=dataset_type,
    data_root=data_root,
    metainfo=metainfo,
    ann_file='/media/fishyu/fish-14tb-2/YiFei/Grounding_DINO/mmdetection/configs/underwater_grounding_dino/oag_train/annotations/rgbs20_val_paired.json',
    data_prefix=dict(img='Images/'),
    pipeline=test_pipeline,
    return_classes=True,
    backend_args=None)

# Bind the RGBS20 datasets explicitly; inherited dataloaders retain RGBS50 otherwise.
train_dataloader = dict(dataset=train_dataset)
val_dataloader = dict(dataset=val_dataset)
test_dataloader = dict(dataset=val_dataset)

val_evaluator = dict(
    type='CocoMetric',
    ann_file='/media/fishyu/fish-14tb-2/YiFei/Grounding_DINO/mmdetection/configs/underwater_grounding_dino/oag_train/annotations/rgbs20_val_paired.json',
    metric='bbox',
    format_only=False,
    backend_args=None)

test_evaluator = val_evaluator