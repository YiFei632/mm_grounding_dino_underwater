from typing import Tuple
import torch
from torch import Tensor
from mmdet.registry import MODELS
from .two_stage import TwoStageDetector

@MODELS.register_module()
class UAMFDet(TwoStageDetector):
    """UAMFDet with independent optical/sonar branches and MDSFF/MIFM."""
    def __init__(self, backbone, rpn_head, roi_head, train_cfg, test_cfg,
                 neck=None, fusion=None, data_preprocessor=None, init_cfg=None):
        super().__init__(backbone=backbone, neck=neck, rpn_head=rpn_head,
                         roi_head=roi_head, train_cfg=train_cfg, test_cfg=test_cfg,
                         data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.sonar_backbone = MODELS.build(backbone)
        self.sonar_neck = MODELS.build(neck) if neck is not None else None
        fusion = fusion or {}
        self.mdsff = MODELS.build(fusion.get('mdsff', dict(type='MDSFF', channels=256)))
        self.mifm = MODELS.build(fusion.get('mifm', dict(type='MIFM', channels=256)))

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        if batch_inputs.shape[1] < 6:
            raise ValueError('UAMFDet expects concatenated optical+sonar input with 6 channels')
        optical, sonar = batch_inputs[:, :3], batch_inputs[:, 3:6]
        ref = self.backbone(optical)
        sensed = self.sonar_backbone(sonar)
        if self.with_neck:
            ref, sensed = self.neck(ref), self.sonar_neck(sensed)
        fused = tuple(self.mdsff(s, r) for s, r in zip(sensed, ref))
        # Apply instance matching on the coarsest feature map and preserve all FPN levels.
        fused = list(fused)
        fused[-1] = self.mifm(fused[-1], ref[-1])
        return tuple(fused)
