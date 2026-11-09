"""Single-modality stages with the exact OAGRGBPrompt checkpoint layout."""
import torch
import torch.nn.functional as F
from torchvision.ops import roi_align

from mmdet.registry import MODELS
from .grounding_dino import GroundingDINO
from .oag_rgb_prompt import OAGRGBPrompt
from ..utils.sonar_rgb_prompt import box_class_targets, spatial_bce_dice


@MODELS.register_module()
class OAGStaged(OAGRGBPrompt):
    def __init__(self, training_stage='rgb', loss_roi_text_weight=1.0, **kwargs):
        if training_stage not in ('rgb', 'sonar', 'fusion'):
            raise ValueError('training_stage must be rgb, sonar or fusion')
        self.training_stage = training_stage
        super().__init__(**kwargs)
        # This loss has no parameters/buffers, preserving checkpoint layout.
        self.roi_text_loss = MODELS.build(dict(
            type='RoIContrastiveFocalLoss', temperature=self.prompt_temperature,
            loss_weight=loss_roi_text_weight))
        if training_stage != 'fusion':
            for name, parameter in self.named_parameters():
                if training_stage == 'rgb':
                    active = not (name.startswith(('sonar_', 'rgb_prompt_mapper.',
                                                    'contrastive_loss_fn.')) or
                                  '.rgb_prompt_attn.' in name)
                else:
                    active = name.startswith((
                        'sonar_backbone.', 'sonar_feat_map.', 'language_model.',
                        'text_feat_map.', 'rgb_prompt_mapper.sonar_features.',
                        'rgb_prompt_mapper.sonar_class.',
                        'rgb_prompt_mapper.text_class.',
                        'rgb_prompt_mapper.class_bias'))
                if not active:
                    parameter.requires_grad_(False)

    def pre_transformer(self, mlvl_feats, batch_data_samples=None):
        if self.training_stage == 'rgb':
            return GroundingDINO.pre_transformer(self, mlvl_feats, batch_data_samples)
        return super().pre_transformer(mlvl_feats, batch_data_samples)

    def forward_encoder(self, *args, **kwargs):
        if self.training_stage == 'rgb':
            kwargs.update(sonar_feat=None, sonar_mask=None, sonar_pos=None)
        return super().forward_encoder(*args, **kwargs)

    def pre_decoder(self, *args, **kwargs):
        if self.training_stage != 'rgb':
            return super().pre_decoder(*args, **kwargs)
        # Use the ordinary language-guided selector with no sonar dependency.
        from .sonarbind_prism import SonarBindPrism
        kwargs.pop('memory_sonar', None)
        kwargs.pop('memory_sonar_mask', None)
        return SonarBindPrism.pre_decoder(self, *args, **kwargs)

    def _compute_contrastive_losses(self, inputs):
        if self.training_stage == 'rgb':
            return {}
        return super()._compute_contrastive_losses(inputs)

    def loss(self, batch_inputs, batch_data_samples):
        if self.training_stage == 'rgb':
            from .sonarbind_prism import SonarBindPrism
            return SonarBindPrism.loss(self, batch_inputs, batch_data_samples)
        if self.training_stage == 'fusion':
            return super().loss(batch_inputs, batch_data_samples)
        return self._sonar_loss(batch_inputs, batch_data_samples)

    def _sonar_loss(self, inputs, samples):
        # The pipeline supplies a fixed sonar canvas and synchronized GT boxes.
        feature = self.sonar_backbone(inputs)[-1]
        if self.fft_band0_filter is not None:
            feature = self.fft_band0_filter(feature)
        mapped = self.sonar_feat_map(feature.flatten(2).transpose(1, 2))
        b, _, h, w = feature.shape
        sonar = self.rgb_prompt_mapper.sonar_features(
            mapped.transpose(1, 2).reshape(b, self.embed_dims, h, w))
        tokens = sonar.flatten(2).transpose(1, 2)
        heat_losses, roi_losses = [], []
        for i, sample in enumerate(samples):
            names = sample.text
            if not isinstance(names, (tuple, list)) or not names:
                raise ValueError('Sonar stage requires return_classes=True')
            caption = '. '.join(names) + '.'
            text = self.language_model([caption])
            text['embedded'] = self.text_feat_map(text['embedded'])
            maps, valid = self._class_token_maps(
                [sample], text['text_token_mask'], text['embedded'].dtype)
            classes = maps @ text['embedded']
            sc = F.normalize(self.rgb_prompt_mapper.sonar_class(tokens[i:i+1]).float(), dim=-1)
            tc = F.normalize(self.rgb_prompt_mapper.text_class(classes).float(), dim=-1)
            logits = sc @ tc.transpose(1, 2) / self.prompt_temperature
            logits = logits + self.rgb_prompt_mapper.class_bias
            target = box_class_targets(sample.gt_instances, inputs.shape[-2:],
                                       (h, w), len(names), inputs.device)
            heat_losses.append(spatial_bce_dice(
                logits, target[None], valid[:, None], logits=True))
            boxes = sample.gt_instances.bboxes
            if hasattr(boxes, 'tensor'):
                boxes = boxes.tensor
            if len(boxes):
                boxes = boxes.clone().to(sonar)
                boxes[:, 0::2] *= w / inputs.shape[-1]
                boxes[:, 1::2] *= h / inputs.shape[-2]
                rois = torch.cat([boxes.new_zeros((len(boxes), 1)), boxes], -1)
                regions = roi_align(sonar[i:i+1], rois, 7, spatial_scale=1.,
                                    sampling_ratio=2, aligned=True).mean((2, 3))
                similarity = F.normalize(regions.float(), dim=-1) @ tc[0].T
                target_roi = F.one_hot(sample.gt_instances.labels.long(), len(names)).float()
                roi_losses.append(self.roi_text_loss(similarity, target_roi))
        return dict(
            loss_sonar_class=self.loss_sonar_class_weight * torch.stack(heat_losses).mean(),
            loss_sonar_text=torch.stack(roi_losses).mean() if roi_losses else tokens.sum() * 0.)
