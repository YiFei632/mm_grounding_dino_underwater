# Base OAG with affine + MLP-residual sonar positions in query selection only.
# The original decoder, dataset, losses, optimizer and schedule are inherited.
_base_ = ['./oag_double_resnet50_rgbs50.py']

custom_imports = dict(
    imports=[
        'mmdet.datasets',
        'mmdet.models',
        'mmdet.models.detectors.oag_affine_query_only',
    ],
    allow_failed_imports=False)

model = dict(
    type='OAGAffineQueryOnly',
    mapping_loss_weight=1.0,
    mapping_residual_limit=0.25,
    query_position_weight=0.5,
    position_prior_sigma=0.1,
)
