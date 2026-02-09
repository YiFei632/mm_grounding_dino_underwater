import torch
import torch.nn.functional as F
from typing import Dict, Tuple
from torch import Tensor
from mmdet.registry import MODELS
from mmdet.utils import OptConfigType
from mmdet.models.detectors.deformable_detr import DeformableDETR

@MODELS.register_module()
class DQDETR(DeformableDETR):
    def __init__(self, transformer=None, **kwargs):
        self.dq_transformer_cfg = transformer
        
        # 1. 清洗配置
        if 'positional_encoding' in kwargs and isinstance(kwargs['positional_encoding'], dict):
            kwargs['positional_encoding'].pop('type', None)
        if 'encoder' in kwargs and isinstance(kwargs['encoder'], dict):
            kwargs['encoder'].pop('type', None)
        if 'decoder' in kwargs and isinstance(kwargs['decoder'], dict):
            kwargs['decoder'].pop('type', None)

        super(DQDETR, self).__init__(**kwargs)

    def _init_layers(self) -> None:
        super()._init_layers()
        
        if self.dq_transformer_cfg is None:
             raise ValueError("DQDETR config requires a 'transformer' dict.")
        
        # 2. 组装自定义 Transformer
        self.transformer = MODELS.build(
            self.dq_transformer_cfg,
            default_args=dict(
                encoder=self.encoder,  
                decoder=self.decoder   
            )
        )
        
        self.embed_dims = self.transformer.embed_dims

        # 3. 【核心修复】删除未使用的参数 (DDP Zombie Params)
        # 我们的 DQTransformer 内部维护了自己的 level_embeds
        if hasattr(self, 'level_embed'):
            del self.level_embed
            
        # DQ-DETR 不使用父类 encoder 
        if hasattr(self, 'encoder'):
            del self.encoder
        if hasattr(self, 'decoder'):
            del self.decoder

        # Two-Stage 模式下，父类会创建这些用于 Query 生成的变换层
        # 但 DQ-DETR 的 Transformer 内部有自己的逻辑，不使用这些
        unused_params = [
            'memory_trans_fc', 'memory_trans_norm', 
            'pos_trans', 'pos_trans_norm',
            'query_embedding' # Two-stage 模式下通常不使用 learnable query
        ]
        
        for param_name in unused_params:
            if hasattr(self, param_name):
                delattr(self, param_name)

    def init_weights(self) -> None:
        if hasattr(self, 'backbone') and self.backbone:
            self.backbone.init_weights()
        if hasattr(self, 'neck') and self.neck:
            self.neck.init_weights()
        if hasattr(self, 'transformer') and self.transformer:
            self.transformer.init_weights()
        if hasattr(self, 'bbox_head') and self.bbox_head:
            self.bbox_head.init_weights()

    def forward_transformer(
        self,
        img_feats: Tuple[Tensor],
        batch_data_samples: OptConfigType = None,
    ) -> Dict:
        mlvl_feats = img_feats
        batch_size = mlvl_feats[0].size(0)
        input_img_h, input_img_w = batch_data_samples[0].batch_input_shape
        
        img_masks = mlvl_feats[0].new_ones(
            (batch_size, input_img_h, input_img_w)).to(torch.bool)
            
        for img_id in range(batch_size):
            img_h, img_w = batch_data_samples[img_id].img_shape
            img_masks[img_id, :img_h, :img_w] = False

        mlvl_masks = []
        mlvl_pos_embeds = []
        
        for feat in mlvl_feats:
            mlvl_masks.append(
                F.interpolate(img_masks[None].float(), size=feat.shape[-2:]).to(torch.bool)[0]
            )
            mlvl_pos_embeds.append(
                self.positional_encoding(mlvl_masks[-1])
            )

        # Query Embedding 在 DQ-DETR Two-Stage 中实际上由 Transformer 内部生成
        # 这里传 None 或者仅用于兼容接口
        query_embed = None

        # 调用自定义 Transformer
        inter_states, init_reference, inter_references, \
        enc_outputs_class, enc_outputs_coord, ccm_logits = self.transformer(
            mlvl_feats=mlvl_feats,
            mlvl_masks=mlvl_masks,
            query_embed=query_embed,
            mlvl_pos_embeds=mlvl_pos_embeds,
            reg_branches=self.bbox_head.reg_branches,
            cls_branches=self.bbox_head.cls_branches
        )

        head_inputs_dict = dict(
            hidden_states=inter_states,
            references=[init_reference, inter_references],
            enc_outputs_class=enc_outputs_class,
            enc_outputs_coord=enc_outputs_coord,
            ccm_logits=ccm_logits 
        )
        
        return head_inputs_dict