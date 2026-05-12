# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.runner.amp import autocast
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList
from .sonarbind import SonarBind

# CLIP 官方归一化参数（RGB 顺序，对应值域 [0, 1]）
# mean = [0.48145466, 0.4578275, 0.40821073]
# std  = [0.26862954, 0.26130258, 0.27577711]
_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
_CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])


@MODELS.register_module()
class CLIPSonarBind(SonarBind):
    """SonarBind with CLIP ViT-L/14 as backbone for all three branches.

    在 SonarBind 基础上，将三个模态的特征提取器全部替换为 CLIP ViT-L/14：

    - 图像分支：CLIPViTBackbone（独立权重 A）
                → ChannelMapper Neck（单输入 1024ch → 4 尺度 256ch）
                → [B, N_img, 256]
    - 声纳分支：CLIPViTBackbone（独立权重 B，与图像分支不共享）
                → sonar_feat_map Linear(1024 → 256)
                → [B, h_p*w_p, 256]
    - 文本分支：CLIPModel（ViT-L/14 文本编码器，hidden_size=768）
                → text_feat_map Linear(768 → 256)
                → [B, N_text, 256]

    Encoder/Decoder（SonarBindTransformerEncoder/Decoder）及两个对比损失
    （loss_sonar_image、loss_sonar_text）完整继承自 SonarBind，不做任何修改。

    与 SonarBind 相比仅有两处改动：
    1. ``_init_layers``：将 sonar_feat_map 的输入维度从 2048 改为 1024
       （CLIP ViT-L/14 hidden_size=1024，ResNet-50 C4 为 2048）。
    2. ``pre_transformer``：声纳图像归一化从 ``/255`` 改为 CLIP mean/std 归一化，
       与图像分支的 data_preprocessor 归一化方式保持一致。
    """

    def _init_layers(self) -> None:
        """复用父类所有层，仅将 sonar_feat_map 输入维度改为 1024。"""
        # 调用 SonarBind._init_layers()，建立全部子模块（包含 Linear(2048→256)）
        super()._init_layers()

        # 覆盖 sonar_feat_map：CLIP ViT-L/14 输出 hidden_size=1024
        self.sonar_feat_map = nn.Linear(1024, self.embed_dims, bias=True)
        nn.init.constant_(self.sonar_feat_map.bias.data, 0)
        nn.init.xavier_uniform_(self.sonar_feat_map.weight.data)

    def pre_transformer(
            self,
            mlvl_feats: Tuple[Tensor],
            batch_data_samples: OptSampleList = None) -> Tuple[Dict, Dict]:
        """处理三路特征，声纳分支改用 CLIP mean/std 归一化。

        图像分支（mlvl_feats）由 extract_feat 经 CLIPViTBackbone + ChannelMapper
        处理，data_preprocessor 已完成 CLIP 归一化，此处直接调用祖父类
        GroundingDINO.pre_transformer 处理即可。

        声纳分支在本方法内重新实现，将原 ``/255`` 归一化替换为 CLIP mean/std
        归一化，再送入 sonar_backbone（CLIPViTBackbone）。
        """
        # ── Step 1: 处理图像分支 ────────────────────────────────────────────
        # 直接调用 GroundingDINO.pre_transformer，跳过 ThirdGroundingDINO 和
        # SonarBind 中附加的声纳处理逻辑，避免重复处理。
        from .grounding_dino import GroundingDINO
        encoder_inputs_dict, decoder_inputs_dict = \
            GroundingDINO.pre_transformer(self, mlvl_feats, batch_data_samples)

        # ── Step 2: 处理声纳分支 ────────────────────────────────────────────
        batch_size = mlvl_feats[0].size(0)
        batch_input_shape = batch_data_samples[0].batch_input_shape
        target_h, target_w = batch_input_shape
        device = mlvl_feats[0].device

        # CLIP 归一化参数，广播形状 [1, 3, 1, 1]
        clip_mean = _CLIP_MEAN.to(device=device, dtype=torch.float32).view(1, 3, 1, 1)
        clip_std = _CLIP_STD.to(device=device, dtype=torch.float32).view(1, 3, 1, 1)

        sonar_imgs_list = []
        for data_samples in batch_data_samples:
            if (not hasattr(data_samples, 'sonar_img')
                    or data_samples.sonar_img is None):
                # 无声纳图像：用全零占位（已处于 CLIP 归一化空间）
                sonar_img = torch.zeros(
                    (3, target_h, target_w), dtype=torch.float32, device=device)
                # 对零图像同样做 CLIP 归一化，保持统计量一致
                sonar_img = (sonar_img - clip_mean.squeeze(0)) / clip_std.squeeze(0)
            else:
                sonar_img = data_samples.sonar_img
                if not isinstance(sonar_img, torch.Tensor):
                    sonar_img = torch.from_numpy(sonar_img)
                sonar_img = sonar_img.to(device=device)
                if sonar_img.dtype == torch.uint8:
                    sonar_img = sonar_img.float()
                # Step A：归一化到 [0, 1]
                if sonar_img.max() > 1.0:
                    sonar_img = sonar_img / 255.0
                # Step B：CLIP mean/std 归一化（对应 data_preprocessor 的处理方式）
                # sonar_img: [3, H, W]，clip_mean/std squeeze 后为 [3, 1, 1]
                sonar_img = (sonar_img - clip_mean.squeeze(0)) / clip_std.squeeze(0)
                # Step C：必要时 resize 对齐 batch 内尺寸
                # （CLIPViTBackbone 内部会再次 resize 到 img_size，此处仅用于 mask 生成）
                if sonar_img.shape[1:] != (target_h, target_w):
                    sonar_img = F.interpolate(
                        sonar_img.unsqueeze(0),
                        size=(target_h, target_w),
                        mode='bilinear',
                        align_corners=False).squeeze(0)
            sonar_imgs_list.append(sonar_img)

        sonar_inputs = torch.stack(sonar_imgs_list)  # [B, 3, target_h, target_w]

        # 通过声纳 CLIPViTBackbone（forward 返回 tuple，取最后一个元素）
        if self.use_autocast:
            with autocast(enabled=True):
                sonar_feats = self.sonar_backbone(sonar_inputs)
        else:
            sonar_feats = self.sonar_backbone(sonar_inputs)

        if isinstance(sonar_feats, (tuple, list)):
            sonar_feat = sonar_feats[-1]   # [B, 1024, h_p, w_p]
        else:
            sonar_feat = sonar_feats

        # 展平 + 线性映射 1024 → 256
        bs, c_s, h_s, w_s = sonar_feat.shape
        sonar_feat_flat = sonar_feat.view(bs, c_s, -1).permute(0, 2, 1)
        # [B, h_s*w_s, 1024] → [B, h_s*w_s, 256]
        sonar_feat_mapped = self.sonar_feat_map(sonar_feat_flat)

        # ── Step 3: 生成声纳位置编码（与 SonarBind 逻辑完全相同） ──────────
        input_img_h, input_img_w = batch_input_shape
        img_shape_list = [sample.img_shape for sample in batch_data_samples]
        same_shape_flag = all(
            s[0] == input_img_h and s[1] == input_img_w
            for s in img_shape_list)

        if torch.onnx.is_in_onnx_export() or same_shape_flag:
            sonar_mask = None
            sonar_pos_embed = self.positional_encoding(
                None,
                input=sonar_feat_mapped.permute(0, 2, 1).view(bs, -1, h_s, w_s))
        else:
            sonar_shape_list = []
            for sample in batch_data_samples:
                if hasattr(sample, 'sonar_shape'):
                    sonar_shape_list.append(sample.sonar_shape)
                elif (hasattr(sample, 'metainfo')
                      and 'sonar_shape' in sample.metainfo):
                    sonar_shape_list.append(sample.metainfo['sonar_shape'])
                else:
                    sonar_shape_list.append(sample.img_shape)

            masks = mlvl_feats[0].new_ones(
                (batch_size, input_img_h, input_img_w))
            for img_id in range(batch_size):
                s_h, s_w = sonar_shape_list[img_id]
                masks[img_id, :s_h, :s_w] = 0

            sonar_mask = F.interpolate(
                masks[None], size=(h_s, w_s)).to(torch.bool).squeeze(0)
            sonar_pos_embed = self.positional_encoding(sonar_mask)
            sonar_mask = sonar_mask.flatten(1)

        sonar_pos_embed = sonar_pos_embed.view(
            bs, -1, h_s * w_s).permute(0, 2, 1)

        encoder_inputs_dict['sonar_feat'] = sonar_feat_mapped
        encoder_inputs_dict['sonar_mask'] = sonar_mask
        encoder_inputs_dict['sonar_pos'] = sonar_pos_embed

        return encoder_inputs_dict, decoder_inputs_dict
