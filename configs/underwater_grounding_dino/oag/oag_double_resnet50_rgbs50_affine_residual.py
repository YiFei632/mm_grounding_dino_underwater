# Learned canonical sonar -> RGB coordinates, with independent paired-GT loss.
_base_ = ['./oag_double_resnet50_rgbs50_position_only_decoder.py']
custom_imports = dict(
    imports=['mmdet.datasets', 'mmdet.models', 'mmdet.models.detectors.oag_affine'],
    allow_failed_imports=False)
model = dict(
    type='OAGAffine',
    loss_sonar_select_weight=0.25,
    mapping_loss_weight=0.25,
    mapping_residual_limit=0.25,
    query_position_weight=0.25,
    position_prior_sigma=0.1,  # Fixed spatial bandwidth, not learned uncertainty.
)
# FairScale encoder checkpoints use reentrant backward. Static DDP supports
# these with unused parameters; empty mapping supervision keeps a zero graph.
model_wrapper_cfg = dict(static_graph=True)

# Keep the 500-iteration warmup
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=500),
]
