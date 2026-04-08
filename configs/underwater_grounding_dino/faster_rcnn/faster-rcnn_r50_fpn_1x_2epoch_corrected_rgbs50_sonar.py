_base_ = [
    '../../_base_/models/faster-rcnn_r50_fpn.py',
    '../../_base_/datasets/rgbs50_sonar_detection.py',
    '../../_base_/schedules/schedule_1x.py', '../../_base_/default_runtime.py'
]
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=2, val_interval=1)