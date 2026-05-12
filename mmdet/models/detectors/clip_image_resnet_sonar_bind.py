# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from mmengine.runner.amp import autocast
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList
from .clip_sonarbind import CLIPSonarBind

# ResNet50 / ImageNet 归一化参数（RGB 顺序，值域 [0, 1]）
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])


@MODELS.register_module()
class CLIPImageResNetSonarBind(CLIPSonarBind):
    """CLIPSonarBind 的变体：图像分支保留 CLIP ViT-L/14，
    声纳分支替换为 ResNet-50。

    与 CLIPSonarBind 相比仅有两处改动：
    1. ``_init_layers``：将 sonar_feat_map 输入维度从 1024 改为 2048
       （ResNet-50 C4 输出通道数为 2048，CLIPViT-L/14 为 1024）。
    2. ``pre_transformer``：声纳图像归一化从 CLIP mean/std 改为
       ImageNet mean/std，并适配 ResNet 多尺度输出格式（取最后一个尺度）。

    其余所有组件（图像 backbone、language_model、neck、encoder、decoder、
    对比损失）完整继承自 CLIPSonarBind，不做任何修改。
    """

    def _init_layers(self) -> None:
        """复用父类所有层，仅将 sonar_feat_map 输入维度改为 2048。"""
        # 调用 CLIPSonarBind._init_layers()，建立全部子模块
        # （其中会建立 sonar_feat_map = Linear(1024→256)）
        super()._init_layers()

        # 覆盖 sonar_feat_map：ResNet-50 C4 输出 2048 通道
        self.sonar_feat_map = nn.Linear(2048, self.embed_dims, bias=True)
        nn.init.constant_(self.sonar_feat_map.bias.data, 0)
        nn.init.xavier_uniform_(self.sonar_feat_map.weight.data)

    def pre_transformer(
            self,
            mlvl_feats: Tuple[Tensor],
            batch_data_samples: OptSampleList = None) -> Tuple[Dict, Dict]:
        """处理三路特征，声纳分支改用 ImageNet mean/std 归一化 + ResNet50。

        图像分支（mlvl_feats）由 extract_feat 经 CLIPViTBackbone + ChannelMapper
        处理，此处直接调用祖父类 GroundingDINO.pre_transformer 处理即可。

        声纳分支在本方法内重新实现：
          - 归一化从 CLIP mean/std 替换为 ImageNet mean/std
          - backbone 输出为多尺度 tuple，取最后一个（C4，2048ch）
          - 线性映射 2048 → 256
        """
        # ── Step 1: 图像分支（跳过 SonarBind/CLIPSonarBind 附加逻辑） ──────
        from .grounding_dino import GroundingDINO
        encoder_inputs_dict, decoder_inputs_dict = \
            GroundingDINO.pre_transformer(self, mlvl_feats, batch_data_samples)

        # ── Step 2: 声纳分支，使用 ImageNet 归一化 ────────────────────────
        batch_size = mlvl_feats[0].size(0)
        batch_input_shape = batch_data_samples[0].batch_input_shape
        target_h, target_w = batch_input_shape
        device = mlvl_feats[0].device

        # ImageNet 归一化参数，广播形状 [1, 3, 1, 1]
        imagenet_mean = _IMAGENET_MEAN.to(
            device=device, dtype=torch.float32).view(1, 3, 1, 1)
        imagenet_std = _IMAGENET_STD.to(
            device=device, dtype=torch.float32).view(1, 3, 1, 1)

        sonar_imgs_list = []
        for data_samples in batch_data_samples:
            if (not hasattr(data_samples, 'sonar_img')
                    or data_samples.sonar_img is None):
                # 无声纳图像：零图像同样做 ImageNet 归一化，保持统计量一致
                sonar_img = torch.zeros(
                    (3, target_h, target_w), dtype=torch.float32, device=device)
                sonar_img = (sonar_img - imagenet_mean.squeeze(0)) / \
                    imagenet_std.squeeze(0)
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
                # Step B：ImageNet mean/std 归一化
                sonar_img = (sonar_img - imagenet_mean.squeeze(0)) / \
                    imagenet_std.squeeze(0)
                # Step C：必要时 resize 对齐 batch 内尺寸
                if sonar_img.shape[1:] != (target_h, target_w):
                    sonar_img = F.interpolate(
                        sonar_img.unsqueeze(0),
                        size=(target_h, target_w),
                        mode='bilinear',
                        align_corners=False).squeeze(0)
            sonar_imgs_list.append(sonar_img)

        sonar_inputs = torch.stack(sonar_imgs_list)  # [B, 3, target_h, target_w]

        # ── Step 3: 通过 ResNet50 sonar_backbone ──────────────────────────
        # ResNet 返回多尺度 tuple：(C2, C3, C4, C5) 或按 out_indices 截取
        # out_indices=(3,) 时只返回 C4：[B, 2048, H/32, W/32]
        if self.use_autocast:
            with autocast(enabled=True):
                sonar_feats = self.sonar_backbone(sonar_inputs)
        else:
            sonar_feats = self.sonar_backbone(sonar_inputs)

        # 取最后一个尺度（C4）
        if isinstance(sonar_feats, (tuple, list)):
            sonar_feat = sonar_feats[-1]   # [B, 2048, H/32, W/32]
        else:
            sonar_feat = sonar_feats

        # 展平 + 线性映射 2048 → 256
        bs, c_s, h_s, w_s = sonar_feat.shape
        sonar_feat_flat = sonar_feat.view(bs, c_s, -1).permute(0, 2, 1)
        # [B, h_s*w_s, 2048] → [B, h_s*w_s, 256]
        sonar_feat_mapped = self.sonar_feat_map(sonar_feat_flat)

        # ── Step 4: 生成声纳位置编码（与 CLIPSonarBind 逻辑完全相同） ──────
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
