_base_ = ['../00_baseline/rgbs50_finetune_20e.py']
# FFTBand0Filter becomes the identity; no FFT/IFFT executes.
model = dict(fft_band0_filter=dict(enabled=False))
work_dir = './work_dirs/oag_ablation/03_no_fft/rgbs50_finetune_20e'
