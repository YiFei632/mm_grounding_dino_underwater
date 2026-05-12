# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from mmdet.registry import MODELS


@MODELS.register_module()
class CLIPViTBackbone(BaseModule):
    """CLIP ViT visual backbone for object detection.

    使用 HuggingFace CLIPVisionModel 提取 patch tokens 作为空间特征图，
    供 ChannelMapper Neck 生成多尺度 FPN 特征。

    与现有 mmdetection backbone 接口完全兼容：forward 返回 tuple，
    tuple 中每个元素为一个空间特征图 [B, C, H, W]。

    Args:
        model_name (str): HuggingFace 模型名或本地权重路径。
            例如 '/path/to/clip-vit-large-patch14'
        img_size (int): backbone 内部统一 resize 的正方形尺寸。
            须为 patch_size(14) 的整数倍。默认 448（448/14=32 → 1024 tokens）。
        frozen_stages (int): 冻结 ViT transformer 层数（从第 0 层起）。
            0  → 不冻结任何 transformer 层（但 patch embedding 始终不冻结）。
            24 → 冻结全部（ViT-L/14 共 24 层）。
            默认 0。
        use_checkpoint (bool): 是否开启 HuggingFace gradient checkpointing
            以降低显存占用。默认 False。
        init_cfg: mmengine 标准初始化配置。权重由 model_name 的
            from_pretrained 自动加载，通常设为 None。
    """

    def __init__(self,
                 model_name: str,
                 img_size: int = 448,
                 frozen_stages: int = 0,
                 use_checkpoint: bool = False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.model_name = model_name
        self.img_size = img_size
        self.frozen_stages = frozen_stages
        self.use_checkpoint = use_checkpoint

        # 延迟导入，避免在不需要时引入 transformers 依赖
        from transformers import CLIPVisionModel, CLIPVisionConfig

        config = CLIPVisionConfig.from_pretrained(model_name)
        self.clip_vision = CLIPVisionModel.from_pretrained(
            model_name, config=config)

        self.patch_size = config.patch_size      # ViT-L/14 → 14
        self.hidden_size = config.hidden_size    # ViT-L/14 → 1024
        self.h_p = img_size // self.patch_size   # 448 // 14 = 32
        self.w_p = img_size // self.patch_size   # 32
        self.num_patches = self.h_p * self.w_p   # 1024

        self._freeze_stages()

        if use_checkpoint:
            self.clip_vision.gradient_checkpointing_enable()

    def _freeze_stages(self):
        """冻结 patch embedding 以及前 frozen_stages 个 transformer 层。"""
        if self.frozen_stages < 0:
            return
        vision_model = self.clip_vision.vision_model
        # 冻结 patch/class/position embedding
        for param in vision_model.embeddings.parameters():
            param.requires_grad = False
        # 逐层冻结
        for i, layer in enumerate(vision_model.encoder.layers):
            if i < self.frozen_stages:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, x: torch.Tensor):
        """Extract CLIP patch token features as spatial feature map.

        Args:
            x (Tensor): [B, 3, H, W]，已完成 CLIP 归一化的图像张量。
                若 H != img_size 或 W != img_size，内部自动双线性插值。

        Returns:
            tuple[Tensor]: 单元素 tuple，元素形状 [B, hidden_size, h_p, w_p]。
                ViT-L/14 + img_size=448 → (Tensor[B, 1024, 32, 32],)
        """
        B, C, H, W = x.shape

        # 统一 resize 至 img_size × img_size
        if H != self.img_size or W != self.img_size:
            x = F.interpolate(
                x,
                size=(self.img_size, self.img_size),
                mode='bilinear',
                align_corners=False)

        # HuggingFace CLIPVisionModel forward
        # last_hidden_state: [B, 1 + h_p*w_p, hidden_size]
        outputs = self.clip_vision(
            pixel_values=x,
            output_hidden_states=False,
            return_dict=True,
            interpolate_pos_encoding=True)

        # 去掉 index=0 的 CLS token，保留全部 patch tokens
        patch_tokens = outputs.last_hidden_state[:, 1:, :]
        # [B, h_p*w_p, hidden_size] → [B, hidden_size, h_p, w_p]
        patch_tokens = patch_tokens.permute(0, 2, 1).reshape(
            B, self.hidden_size, self.h_p, self.w_p)

        return (patch_tokens,)
