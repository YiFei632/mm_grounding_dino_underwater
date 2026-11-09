_base_ = ['./_base_staged.py']
model = dict(training_stage='fusion',loss_sonar_image_weight=0, bbox_head=dict(num_classes=6))
work_dir = './work_dirs/oag_train/single_fusion_rgbs50'
# Loads RGB detector + sonar specialist + shared text weights together.
load_from = '/media/fishyu/fish-14tb-2/YiFei/Grounding_DINO/mmdetection/work_dirs/oag_train/sonar_sctd_coco/epoch_12.pth'

train_dataloader = dict(batch_size=4)
val_dataloader = dict(batch_size=4)