# RGBS20 fine-tuning with the no-FFT ablation.
_base_ = [
    '../../oag_train/oag_double_resnet50_single_fusion_rgbs20_finetune_12e.py'
]

# Keep the RGBS20 dataset, paired sonar annotations, and 12-epoch schedule.
model = dict(fft_band0_filter=dict(enabled=False))
work_dir = './work_dirs/oag_ablation/03_no_fft/rgbs20_finetune_12e'
