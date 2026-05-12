# Copyright (c) OpenMMLab. All rights reserved.
import copy
import re
import warnings
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.runner.amp import autocast
from torch import Tensor
# from torch.profiler import ProfilerActivity, profile, record_function

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType
from ..layers import SinePositionalEncoding
from ..layers.transformer.sonarbind_layers import (
    SonarBindTransformerDecoder, SonarBindTransformerEncoder)
from ..losses import ContrastiveLoss
from .third_grounding_dino import ThirdGroundingDINO
from .glip import (create_positive_map, create_positive_map_label_to_token,
                   run_ner)


def clean_label_name(name: str) -> str:
    name = re.sub(r'\(.*\)', '', name)
    name = re.sub(r'_', ' ', name)
    name = re.sub(r'  ', ' ', name)
    return name


def chunks(lst: list, n: int) -> list:
    """Yield successive n-sized chunks from lst."""
    all_ = []
    for i in range(0, len(lst), n):
        data_index = lst[i:i + n]
        all_.append(data_index)
    counter = 0
    for i in all_:
        counter += len(i)
    assert (counter == len(lst))

    return all_


@MODELS.register_module()
class SonarBindLGANet(ThirdGroundingDINO):
    """SonarBind variant that uses LGANet as the sonar backbone.

    Identical to SonarBind except that ``sonar_feat_channels`` is exposed as
    an explicit constructor argument so that the projection layer
    ``sonar_feat_map`` can be sized correctly for backbones whose final-stage
    output channel count differs from ResNet-50's 2048 (e.g. LGANet outputs
    512 channels at P5).

    Original class: SonarBind.
    See also: `Grounding DINO: Marrying DINO with Grounded Pre-
    Training for Open-Set Object Detection.

    <https://arxiv.org/abs/2303.05499>`_

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/GroundingDINO>`_.
    """

    def __init__(self,
                 language_model,
                 sonar_backbone,
                 *args,
                 use_autocast=False,
                 sonar_feat_channels=512,
                 loss_sonar_image_weight=1.0,
                 loss_sonar_text_weight=1.0,
                 contrastive_temperature=0.07,
                 learnable_temperature=False,
                 **kwargs) -> None:

        self.language_model_cfg = language_model
        self.sonar_backbone_cfg = sonar_backbone
        self._special_tokens = '. '
        self.use_autocast = use_autocast
        self.sonar_feat_channels = sonar_feat_channels

        # Contrastive loss configuration
        self.loss_sonar_image_weight = loss_sonar_image_weight
        self.loss_sonar_text_weight = loss_sonar_text_weight
        self.contrastive_temperature = contrastive_temperature
        self.learnable_temperature = learnable_temperature

        # 将language_model传递给父类
        super().__init__(language_model=language_model, sonar_backbone=sonar_backbone, *args, **kwargs)

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        self.encoder = SonarBindTransformerEncoder(**self.encoder)
        self.decoder = SonarBindTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)
        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            f'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'

        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)

        # text modules
        self.language_model = MODELS.build(self.language_model_cfg)
        self.sonar_backbone = MODELS.build(self.sonar_backbone_cfg)
        self.text_feat_map = nn.Linear(
            self.language_model.language_backbone.body.language_dim,
            self.embed_dims,
            bias=True)
        
        self.sonar_feat_map = nn.Linear(
            self.sonar_feat_channels,
            self.embed_dims,
            bias=True)

        # Contrastive loss module for multi-modal alignment
        self.contrastive_loss_fn = ContrastiveLoss(
            temperature=self.contrastive_temperature,
            learnable_temperature=self.learnable_temperature
        )

    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super().init_weights()
        nn.init.constant_(self.text_feat_map.bias.data, 0)
        nn.init.xavier_uniform_(self.text_feat_map.weight.data)
        nn.init.constant_(self.sonar_feat_map.bias.data, 0)
        nn.init.xavier_uniform_(self.sonar_feat_map.weight.data)

    def to_enhance_text_prompts(self, original_caption, enhanced_text_prompts):
        caption_string = ''
        tokens_positive = []
        for idx, word in enumerate(original_caption):
            if word in enhanced_text_prompts:
                enhanced_text_dict = enhanced_text_prompts[word]
                if 'prefix' in enhanced_text_dict:
                    caption_string += enhanced_text_dict['prefix']
                start_i = len(caption_string)
                if 'name' in enhanced_text_dict:
                    caption_string += enhanced_text_dict['name']
                else:
                    caption_string += word
                end_i = len(caption_string)
                tokens_positive.append([[start_i, end_i]])

                if 'suffix' in enhanced_text_dict:
                    caption_string += enhanced_text_dict['suffix']
            else:
                tokens_positive.append(
                    [[len(caption_string),
                      len(caption_string) + len(word)]])
                caption_string += word
            caption_string += self._special_tokens
        return caption_string, tokens_positive

    def to_plain_text_prompts(self, original_caption):
        caption_string = ''
        tokens_positive = []
        for idx, word in enumerate(original_caption):
            tokens_positive.append(
                [[len(caption_string),
                  len(caption_string) + len(word)]])
            caption_string += word
            caption_string += self._special_tokens
        return caption_string, tokens_positive

    def get_tokens_and_prompts(
        self,
        original_caption: Union[str, list, tuple],
        custom_entities: bool = False,
        enhanced_text_prompts: Optional[ConfigType] = None
    ) -> Tuple[dict, str, list]:
        """Get the tokens positive and prompts for the caption."""
        if isinstance(original_caption, (list, tuple)) or custom_entities:
            if custom_entities and isinstance(original_caption, str):
                original_caption = original_caption.strip(self._special_tokens)
                original_caption = original_caption.split(self._special_tokens)
                original_caption = list(
                    filter(lambda x: len(x) > 0, original_caption))

            original_caption = [clean_label_name(i) for i in original_caption]

            if custom_entities and enhanced_text_prompts is not None:
                caption_string, tokens_positive = self.to_enhance_text_prompts(
                    original_caption, enhanced_text_prompts)
            else:
                caption_string, tokens_positive = self.to_plain_text_prompts(
                    original_caption)

            # NOTE: Tokenizer in Grounding DINO is different from
            # that in GLIP. The tokenizer in GLIP will pad the
            # caption_string to max_length, while the tokenizer
            # in Grounding DINO will not.
            tokenized = self.language_model.tokenizer(
                [caption_string],
                padding='max_length'
                if self.language_model.pad_to_max else 'longest',
                return_tensors='pt')
            entities = original_caption
        else:
            if not original_caption.endswith('.'):
                original_caption = original_caption + self._special_tokens
            # NOTE: Tokenizer in Grounding DINO is different from
            # that in GLIP. The tokenizer in GLIP will pad the
            # caption_string to max_length, while the tokenizer
            # in Grounding DINO will not.
            tokenized = self.language_model.tokenizer(
                [original_caption],
                padding='max_length'
                if self.language_model.pad_to_max else 'longest',
                return_tensors='pt')
            tokens_positive, noun_phrases = run_ner(original_caption)
            entities = noun_phrases
            caption_string = original_caption

        return tokenized, caption_string, tokens_positive, entities

    def get_positive_map(self, tokenized, tokens_positive):
        positive_map = create_positive_map(
            tokenized,
            tokens_positive,
            max_num_entities=self.bbox_head.cls_branches[
                self.decoder.num_layers].max_text_len)
        positive_map_label_to_token = create_positive_map_label_to_token(
            positive_map, plus=1)
        return positive_map_label_to_token, positive_map

    def get_tokens_positive_and_prompts(
        self,
        original_caption: Union[str, list, tuple],
        custom_entities: bool = False,
        enhanced_text_prompt: Optional[ConfigType] = None,
        tokens_positive: Optional[list] = None,
    ) -> Tuple[dict, str, Tensor, list]:
        """Get the tokens positive and prompts for the caption.

        Args:
            original_caption (str): The original caption, e.g. 'bench . car .'
            custom_entities (bool, optional): Whether to use custom entities.
                If ``True``, the ``original_caption`` should be a list of
                strings, each of which is a word. Defaults to False.

        Returns:
            Tuple[dict, str, dict, str]: The dict is a mapping from each entity
            id, which is numbered from 1, to its positive token id.
            The str represents the prompts.
        """
        if tokens_positive is not None:
            if tokens_positive == -1:
                if not original_caption.endswith('.'):
                    original_caption = original_caption + self._special_tokens
                return None, original_caption, None, original_caption
            else:
                if not original_caption.endswith('.'):
                    original_caption = original_caption + self._special_tokens
                tokenized = self.language_model.tokenizer(
                    [original_caption],
                    padding='max_length'
                    if self.language_model.pad_to_max else 'longest',
                    return_tensors='pt')
                positive_map_label_to_token, positive_map = \
                    self.get_positive_map(tokenized, tokens_positive)

                entities = []
                for token_positive in tokens_positive:
                    instance_entities = []
                    for t in token_positive:
                        instance_entities.append(original_caption[t[0]:t[1]])
                    entities.append(' / '.join(instance_entities))
                return positive_map_label_to_token, original_caption, \
                    positive_map, entities

        chunked_size = self.test_cfg.get('chunked_size', -1)
        if not self.training and chunked_size > 0:
            assert isinstance(original_caption,
                              (list, tuple)) or custom_entities is True
            all_output = self.get_tokens_positive_and_prompts_chunked(
                original_caption, enhanced_text_prompt)
            positive_map_label_to_token, \
                caption_string, \
                positive_map, \
                entities = all_output
        else:
            tokenized, caption_string, tokens_positive, entities = \
                self.get_tokens_and_prompts(
                    original_caption, custom_entities, enhanced_text_prompt)
            positive_map_label_to_token, positive_map = self.get_positive_map(
                tokenized, tokens_positive)
        return positive_map_label_to_token, caption_string, \
            positive_map, entities

    def get_tokens_positive_and_prompts_chunked(
            self,
            original_caption: Union[list, tuple],
            enhanced_text_prompts: Optional[ConfigType] = None):
        chunked_size = self.test_cfg.get('chunked_size', -1)
        original_caption = [clean_label_name(i) for i in original_caption]

        original_caption_chunked = chunks(original_caption, chunked_size)
        ids_chunked = chunks(
            list(range(1,
                       len(original_caption) + 1)), chunked_size)

        positive_map_label_to_token_chunked = []
        caption_string_chunked = []
        positive_map_chunked = []
        entities_chunked = []

        for i in range(len(ids_chunked)):
            if enhanced_text_prompts is not None:
                caption_string, tokens_positive = self.to_enhance_text_prompts(
                    original_caption_chunked[i], enhanced_text_prompts)
            else:
                caption_string, tokens_positive = self.to_plain_text_prompts(
                    original_caption_chunked[i])
            tokenized = self.language_model.tokenizer([caption_string],
                                                      return_tensors='pt')
            if tokenized.input_ids.shape[1] > self.language_model.max_tokens:
                warnings.warn('Inputting a text that is too long will result '
                              'in poor prediction performance. '
                              'Please reduce the --chunked-size.')
            positive_map_label_to_token, positive_map = self.get_positive_map(
                tokenized, tokens_positive)

            caption_string_chunked.append(caption_string)
            positive_map_label_to_token_chunked.append(
                positive_map_label_to_token)
            positive_map_chunked.append(positive_map)
            entities_chunked.append(original_caption_chunked[i])

        return positive_map_label_to_token_chunked, \
            caption_string_chunked, \
            positive_map_chunked, \
            entities_chunked

    def pre_transformer(                                                                                                                                                                                                                 
            self,                                                                                                                                                                                                                        
            mlvl_feats: Tuple[Tensor],                                                                                                                                                                                                   
            batch_data_samples: OptSampleList = None) -> Tuple[Dict, Dict]:                                                                                                                                                              
        """Extend parent's pre_transformer to support sonar features."""                                                                                                                                                                 
                                                                                                                                                                                                                                        
        # 1. 调用父类方法处理image特征                                                                                                                                                                                                   
        encoder_inputs_dict, decoder_inputs_dict = super().pre_transformer(                                                                                                                                                              
            mlvl_feats, batch_data_samples)                                                                                                                                                                                              
                                                                                                                                                                                                                                        
        # 2. 提取并处理sonar特征
        batch_size = mlvl_feats[0].size(0)
        device = mlvl_feats[0].device

        # 声呐图像固定 resize 到 224×224，与 RGB backbone 预训练尺寸对齐
        SONAR_SIZE = (224, 224)

        # ImageNet mean/std（与 RGB DetDataPreprocessor 保持一致）
        # mean: [123.675, 116.28, 103.53] / 255  std: [58.395, 57.12, 57.375] / 255
        sonar_mean = torch.tensor([0.485, 0.456, 0.406],
                                  dtype=torch.float32, device=device).view(3, 1, 1)
        sonar_std  = torch.tensor([0.229, 0.224, 0.225],
                                  dtype=torch.float32, device=device).view(3, 1, 1)

        # 提取并预处理sonar图像
        sonar_imgs_list = []
        for data_samples in batch_data_samples:
            if not hasattr(data_samples, 'sonar_img') or data_samples.sonar_img is None:
                # fallback: 全零张量（已归一化空间）
                sonar_img = torch.zeros(
                    (3, *SONAR_SIZE), dtype=torch.float32, device=device)
            else:
                sonar_img = data_samples.sonar_img
                if not isinstance(sonar_img, torch.Tensor):
                    sonar_img = torch.from_numpy(sonar_img)
                if sonar_img.device != device:
                    sonar_img = sonar_img.to(device)

                # uint8 → float32
                if sonar_img.dtype == torch.uint8:
                    sonar_img = sonar_img.float()

                # [0, 255] → [0, 1]
                if sonar_img.max() > 1.0:
                    sonar_img = sonar_img / 255.0

                # Resize 到 224×224
                if sonar_img.shape[1:] != SONAR_SIZE:
                    sonar_img = F.interpolate(
                        sonar_img.unsqueeze(0),
                        size=SONAR_SIZE,
                        mode='bilinear',
                        align_corners=False,
                    ).squeeze(0)

                # ImageNet 归一化：(x - mean) / std
                sonar_img = (sonar_img - sonar_mean) / sonar_std

            sonar_imgs_list.append(sonar_img)

        sonar_inputs = torch.stack(sonar_imgs_list)

        # 通过sonar_backbone                                                                                                                                                                                                             
        if self.use_autocast:                                                                                                                                                                                                            
            with autocast(enabled=True):                                                                                                                                                                                                 
                sonar_feats = self.sonar_backbone(sonar_inputs)                                                                                                                                                                          
        else:                                                                                                                                                                                                                            
            sonar_feats = self.sonar_backbone(sonar_inputs)                                                                                                                                                                              
                                                                                                                                                                                                                                        
        # 处理sonar特征                                                                                                                                                                                                                  
        if isinstance(sonar_feats, (tuple, list)):                                                                                                                                                                                               
            sonar_feat = sonar_feats[-1]                                                                                                                                                                                                 
        else:                                                                                                                                                                                                                            
            sonar_feat = sonar_feats                                                                                                                                                                                                     
                                                                                                                                                                                                                                        
        bs, c_s, h_s, w_s = sonar_feat.shape
        sonar_feat_flat = sonar_feat.view(bs, c_s, -1).permute(0, 2, 1)
        sonar_feat_mapped = self.sonar_feat_map(sonar_feat_flat)                                                                                                                                                                                     
                                                                                                                                                                                                                                        
        # 生成sonar位置编码                                                                                                                                                                                                              
        batch_input_shape = batch_data_samples[0].batch_input_shape                                                                                                                                                                      
        input_img_h, input_img_w = batch_input_shape                                                                                                                                                                                     
        img_shape_list = [sample.img_shape for sample in batch_data_samples]                                                                                                                                                             
        same_shape_flag = all([                                                                                                                                                                                                          
            s[0] == input_img_h and s[1] == input_img_w for s in img_shape_list                                                                                                                                                          
        ])                                                                                                                                                                                                                               
                                                                                                                                                                                                                                        
        if torch.onnx.is_in_onnx_export() or same_shape_flag:
            sonar_mask = None
            sonar_pos_embed = self.positional_encoding(None, input=sonar_feat_mapped.permute(0, 2, 1).view(bs, -1, h_s, w_s))
        else:
            # 获取原始声纳图像尺寸
            sonar_shape_list = []
            for sample in batch_data_samples:
                if hasattr(sample, 'sonar_shape'):
                    sonar_shape_list.append(sample.sonar_shape)
                elif hasattr(sample.metainfo, 'sonar_shape'):
                    sonar_shape_list.append(sample.metainfo['sonar_shape'])
                else:
                    sonar_shape_list.append(sample.img_shape)
            masks = mlvl_feats[0].new_ones((batch_size, input_img_h, input_img_w))                                                                                                                                                       
            for img_id in range(batch_size):                                                                                                                                                                                             
                s_h, s_w = sonar_shape_list[img_id]                                                                                                                                                                                      
                masks[img_id, :s_h, :s_w] = 0                                                                                                                                                                                            
                                                                                                                                                                                                                                        
            sonar_mask = F.interpolate(                                                                                                                                                                                                  
                masks[None], size=(h_s, w_s)).to(torch.bool).squeeze(0)                                                                                                                                                                  
            sonar_pos_embed = self.positional_encoding(sonar_mask)                                                                                                                                                                       
            sonar_mask = sonar_mask.flatten(1)                                                                                                                                                                                           
                                                                                                                                                                                                                                        
        sonar_pos_embed = sonar_pos_embed.view(bs, -1, h_s * w_s).permute(0, 2, 1)                                                                                                                                                       
                                                                                                                                                                                                                                        
        # 3. 将sonar特征添加到encoder_inputs_dict
        encoder_inputs_dict['sonar_feat'] = sonar_feat_mapped
        encoder_inputs_dict['sonar_mask'] = sonar_mask
        encoder_inputs_dict['sonar_pos'] = sonar_pos_embed                                                                                                                                                                               
                                                                                                                                                                                                                                        
        return encoder_inputs_dict, decoder_inputs_dict
    
    def forward_transformer(
        self,
        img_feats: Tuple[Tensor],
        text_dict: Dict,
        batch_data_samples: OptSampleList = None,
    ) -> Dict:
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)
        encoder_outputs_dict = self.forward_encoder(
            **encoder_inputs_dict, text_dict=text_dict)

        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict, batch_data_samples=batch_data_samples)
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)
        return head_inputs_dict

    def forward_encoder(self, feat: Tensor, feat_mask: Tensor,
                        feat_pos: Tensor, sonar_feat: Tensor,
                        sonar_mask: Tensor, sonar_pos: Tensor,
                        spatial_shapes: Tensor,
                        level_start_index: Tensor, valid_ratios: Tensor,
                        text_dict: Dict) -> Dict:
        text_token_mask = text_dict['text_token_mask']
        memory, memory_sonar, memory_text = self.encoder(
            query=feat,
            query_sonar=sonar_feat,
            query_pos=feat_pos,
            query_sonar_pos=sonar_pos,
            key_padding_mask=feat_mask,  # for self_attn
            sonar_padding_mask=sonar_mask,  # for sonar self_attn
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            # for text encoder
            memory_text=text_dict['embedded'],
            text_attention_mask=~text_token_mask,
            position_ids=text_dict['position_ids'],
            text_self_attention_masks=text_dict['masks'])
        encoder_outputs_dict = dict(
            memory=memory,
            memory_sonar=memory_sonar,
            memory_sonar_mask=sonar_mask,
            memory_mask=feat_mask,
            spatial_shapes=spatial_shapes,
            memory_text=memory_text,
            text_token_mask=text_token_mask)
        return encoder_outputs_dict

    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        memory_text: Tensor,
        text_token_mask: Tensor,
        memory_sonar: Tensor = None,
        memory_sonar_mask: Tensor = None,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict]:
        bs, _, c = memory.shape

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)

        enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers](output_memory, memory_text,
                                     text_token_mask)
        cls_out_features = self.bbox_head.cls_branches[
            self.decoder.num_layers].max_text_len
        enc_outputs_coord_unact = self.bbox_head.reg_branches[
            self.decoder.num_layers](output_memory) + output_proposals

        # NOTE The DINO selects top-k proposals according to scores of
        # multi-class classification, while DeformDETR, where the input
        # is `enc_outputs_class[..., 0]` selects according to scores of
        # binary classification.
        topk_indices = torch.topk(
            enc_outputs_class.max(-1)[0], k=self.num_queries, dim=1)[1]

        topk_score = torch.gather(
            enc_outputs_class, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        topk_coords = topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()

        query = self.query_embedding.weight[:, None, :]
        query = query.repeat(1, bs, 1).transpose(0, 1)
        if self.training:
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
                self.dn_query_generator(batch_data_samples)
            query = torch.cat([dn_label_query, query], dim=1)
            reference_points = torch.cat([dn_bbox_query, topk_coords_unact],
                                         dim=1)
        else:
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None
        reference_points = reference_points.sigmoid()

        decoder_inputs_dict = dict(
            query=query,
            memory=memory,
            reference_points=reference_points,
            dn_mask=dn_mask,
            memory_text=memory_text,
            text_attention_mask=~text_token_mask,
        )
        # NOTE DINO calculates encoder losses on scores and coordinates
        # of selected top-k encoder queries, while DeformDETR is of all
        # encoder queries.
        head_inputs_dict = dict(
            enc_outputs_class=topk_score,
            enc_outputs_coord=topk_coords,
            dn_meta=dn_meta) if self.training else dict()
        # append text_feats to head_inputs_dict
        head_inputs_dict['memory_text'] = memory_text
        head_inputs_dict['text_token_mask'] = text_token_mask
        # append sonar_feats to head_inputs_dict (sonar不进decoder，与memory_text对称保存)
        head_inputs_dict['memory_sonar'] = memory_sonar
        head_inputs_dict['memory_sonar_mask'] = memory_sonar_mask
        head_inputs_dict['memory'] = memory                                                                                                                                                                      
        head_inputs_dict['memory_mask'] = memory_mask                                                                                                                                                            
        head_inputs_dict['spatial_shapes'] = spatial_shapes
        return decoder_inputs_dict, head_inputs_dict

    def _compute_contrastive_losses(self, head_inputs_dict: Dict) -> Dict:
        """Compute contrastive losses between sonar and other modalities.

        This method implements CLIP-style contrastive learning to align
        sonar features with image and text features in a shared embedding space.

        Args:
            head_inputs_dict (Dict): Dictionary containing:
                - memory_sonar: Sonar features from encoder, shape (bs, num_sonar_tokens, dim)
                - memory_sonar_mask: Sonar padding mask, shape (bs, num_sonar_tokens)
                - memory: Image features from encoder, shape (bs, num_image_tokens, dim)
                - memory_mask: Image padding mask, shape (bs, num_image_tokens)
                - memory_text: Text features from encoder, shape (bs, num_text_tokens, dim)
                - text_token_mask: Text token mask, shape (bs, num_text_tokens)

        Returns:
            Dict: Dictionary containing:
                - loss_sonar_image: Contrastive loss between sonar and image
                - loss_sonar_text: Contrastive loss between sonar and text
        """
        losses = {}

        # Extract features and masks from head_inputs_dict
        memory_sonar = head_inputs_dict.get('memory_sonar', None)
        memory_sonar_mask = head_inputs_dict.get('memory_sonar_mask', None)
        memory_image = head_inputs_dict.get('memory', None)
        memory_image_mask = head_inputs_dict.get('memory_mask', None)
        memory_text = head_inputs_dict.get('memory_text', None)
        text_token_mask = head_inputs_dict.get('text_token_mask', None)

        if memory_sonar is None:
            # If no sonar features available, return zero losses
            device = memory_image.device if memory_image is not None else torch.device('cpu')
            return {
                'loss_sonar_image': torch.tensor(0.0, device=device),
                'loss_sonar_text': torch.tensor(0.0, device=device)
            }

        # Handle mask polarity:
        # - memory_sonar_mask and memory_image_mask: True for padding, need to invert
        # - text_token_mask: True for valid tokens, already correct polarity
        sonar_mask_valid = ~memory_sonar_mask if memory_sonar_mask is not None else None
        image_mask_valid = ~memory_image_mask if memory_image_mask is not None else None

        # 1. Compute Sonar-Image contrastive loss
        loss_sonar_image = self.contrastive_loss_fn(
            memory_sonar, memory_image,
            mask_a=sonar_mask_valid,
            mask_b=image_mask_valid
        )
        losses['loss_sonar_image'] = loss_sonar_image * self.loss_sonar_image_weight

        # 2. Compute Sonar-Text contrastive loss
        loss_sonar_text = self.contrastive_loss_fn(
            memory_sonar, memory_text,
            mask_a=sonar_mask_valid,
            mask_b=text_token_mask  # Already correct polarity (True for valid)
        )
        losses['loss_sonar_text'] = loss_sonar_text * self.loss_sonar_text_weight

        return losses

    # -------------------------------------------------------------------------
    # Memory profiling helpers (disabled – uncomment + set SONARBIND_PROFILE=1
    # to re-enable one-shot memory tracing)
    # -------------------------------------------------------------------------
    # _profile_done: bool = False  # class-level flag: only profile once

    # @staticmethod
    # def _get_nvml_mem_gb() -> str:
    #     """Query GPU memory via NVML, matching what nvitop displays.
    #
    #     Uses v2 API (nvidia-ml-py >= 12.x) if available, falls back to v1.
    #     - proc=X.XXX GB : per-process memory for THIS process (nvitop process row)
    #     - dev_total=X.XXX GB : total device used memory (nvitop top bar)
    #     """
    #     try:
    #         import pynvml, os
    #         pynvml.nvmlInit()
    #         cuda_dev = torch.cuda.current_device()
    #         handle   = pynvml.nvmlDeviceGetHandleByIndex(cuda_dev)
    #         pid      = os.getpid()
    #
    #         # Total device memory (= nvitop top progress bar)
    #         mem_info  = pynvml.nvmlDeviceGetMemoryInfo(handle)
    #         dev_used  = mem_info.used / 1024 ** 3
    #
    #         # Per-process memory – prefer v2 API (more accurate on driver >= 520)
    #         proc_used = float('nan')
    #         try:
    #             all_procs = pynvml.nvmlDeviceGetComputeRunningProcesses_v2(handle)
    #         except AttributeError:
    #             all_procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    #         # Also include graphics processes (display server, etc.)
    #         try:
    #             all_procs = list(all_procs) + list(
    #                 pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle))
    #         except Exception:
    #             pass
    #         for p in all_procs:
    #             if p.pid == pid:
    #                 if hasattr(p, 'usedGpuMemory') and p.usedGpuMemory:
    #                     proc_used = p.usedGpuMemory / 1024 ** 3
    #                 break
    #
    #         return (f'proc={proc_used:.3f}GB  dev_total={dev_used:.3f}GB')
    #     except Exception as e:
    #         return f'nvml_err({e})'

    # @staticmethod
    # def _mem_snapshot(tag: str) -> None:
    #     torch.cuda.synchronize()
    #     alloc = torch.cuda.memory_allocated() / 1024 ** 3
    #     resv  = torch.cuda.memory_reserved()  / 1024 ** 3
    #     nvml  = SonarBind._get_nvml_mem_gb()
    #     print(f'[MEM] {tag:<45s}  '
    #           f'torch_alloc={alloc:.3f}GB  torch_resv={resv:.3f}GB  '
    #           f'nvml({nvml})')

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        # -- text tokenisation --
        text_prompts = []
        for data_samples in batch_data_samples:
            if hasattr(data_samples, 'text') and data_samples.text is not None:
                if isinstance(data_samples.text, (tuple, list)):
                    clean_classes = [clean_label_name(c) for c in data_samples.text]
                    text_prompts.append('. '.join(clean_classes) + '.')
                else:
                    text_prompts.append(data_samples.text)
            else:
                if hasattr(data_samples, 'metainfo') and 'classes' in data_samples.metainfo:
                    classes = data_samples.metainfo['classes']
                    clean_classes = [clean_label_name(c) for c in classes]
                    text_prompts.append('. '.join(clean_classes) + '.')
                else:
                    raise ValueError(
                        'data_samples must have either text attribute or '
                        'metainfo with classes to generate text prompts')

        gt_labels = [
            data_samples.gt_instances.labels
            for data_samples in batch_data_samples
        ]

        if 'tokens_positive' in batch_data_samples[0]:
            tokens_positive = [
                data_samples.tokens_positive
                for data_samples in batch_data_samples
            ]
            positive_maps = []
            for token_positive, text_prompt, gt_label in zip(
                    tokens_positive, text_prompts, gt_labels):
                tokenized = self.language_model.tokenizer(
                    [text_prompt],
                    padding='max_length'
                    if self.language_model.pad_to_max else 'longest',
                    return_tensors='pt')
                new_tokens_positive = [
                    token_positive[label.item()] for label in gt_label
                ]
                _, positive_map = self.get_positive_map(
                    tokenized, new_tokens_positive)
                positive_maps.append(positive_map)
            new_text_prompts = text_prompts
        else:
            new_text_prompts = []
            positive_maps = []
            if len(set(text_prompts)) == 1:
                tokenized, caption_string, tokens_positive, _ = \
                    self.get_tokens_and_prompts(text_prompts[0], True)
                new_text_prompts = [caption_string] * len(batch_inputs)
                for gt_label in gt_labels:
                    new_tokens_positive = [
                        tokens_positive[label] for label in gt_label
                    ]
                    _, positive_map = self.get_positive_map(
                        tokenized, new_tokens_positive)
                    positive_maps.append(positive_map)
            else:
                for text_prompt, gt_label in zip(text_prompts, gt_labels):
                    tokenized, caption_string, tokens_positive, _ = \
                        self.get_tokens_and_prompts(text_prompt, True)
                    new_tokens_positive = [
                        tokens_positive[label] for label in gt_label
                    ]
                    _, positive_map = self.get_positive_map(
                        tokenized, new_tokens_positive)
                    positive_maps.append(positive_map)
                    new_text_prompts.append(caption_string)

        # -- BERT language model --
        text_dict = self.language_model(new_text_prompts)
        if self.text_feat_map is not None:
            text_dict['embedded'] = self.text_feat_map(text_dict['embedded'])
        for i, data_samples in enumerate(batch_data_samples):
            positive_map = positive_maps[i].to(
                batch_inputs.device).bool().float()
            text_token_mask = text_dict['text_token_mask'][i]
            data_samples.gt_instances.positive_maps = positive_map
            data_samples.gt_instances.text_token_mask = \
                text_token_mask.unsqueeze(0).repeat(len(positive_map), 1)

        # -- RGB backbone + neck --
        if self.use_autocast:
            with autocast(enabled=True):
                visual_features = self.extract_feat(batch_inputs)
        else:
            visual_features = self.extract_feat(batch_inputs)

        # -- sonar backbone + pre_transformer --
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            visual_features, batch_data_samples)

        # -- transformer encoder --
        encoder_outputs_dict = self.forward_encoder(
            **encoder_inputs_dict, text_dict=text_dict)

        # -- pre_decoder (top-k selection + DN queries) --
        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict, batch_data_samples=batch_data_samples)
        decoder_inputs_dict.update(tmp_dec_in)

        # -- transformer decoder --
        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)

        # -- contrastive loss --
        contrastive_losses = self._compute_contrastive_losses(head_inputs_dict)

        # -- detection loss --
        bbox_head_inputs = {
            k: v for k, v in head_inputs_dict.items()
            if k not in ['memory_sonar', 'memory_sonar_mask',
                         'memory', 'memory_mask', 'spatial_shapes']
        }
        losses = self.bbox_head.loss(
            **bbox_head_inputs, batch_data_samples=batch_data_samples)

        losses.update(contrastive_losses)

        return losses

    # def train_step(self, data, optim_wrapper):
    #     """Override train_step to monitor memory across the full training step.
    #
    #     Covers: data-preprocess → forward → backward → optimizer → zero_grad.
    #     Activate with:  SONARBIND_PROFILE=1 python tools/train.py ...
    #     """
    #     import os
    #     _do_profile = (os.environ.get('SONARBIND_PROFILE', '0') == '1'
    #                    and not SonarBind._profile_done)
    #
    #     if not _do_profile:
    #         # Normal fast path – identical to mmengine BaseModel.train_step
    #         with optim_wrapper.optim_context(self):
    #             data = self.data_preprocessor(data, True)
    #             losses = self._run_forward(data, mode='loss')
    #         parsed_losses, log_vars = self.parse_losses(losses)
    #         optim_wrapper.update_params(parsed_losses)
    #         return log_vars
    #
    #     # ── Profiling path ────────────────────────────────────────────────────
    #     SonarBind._profile_done = True          # only profile one step
    #     torch.cuda.reset_peak_memory_stats()
    #     torch.cuda.synchronize()
    #
    #     sep = '=' * 70
    #     print(f'\n{sep}')
    #     print('[SonarBind Profiler] Full training-step memory trace')
    #     print(sep)
    #
    #     def snap(tag):
    #         torch.cuda.synchronize()
    #         alloc = torch.cuda.memory_allocated()     / 1024 ** 3
    #         resv  = torch.cuda.memory_reserved()      / 1024 ** 3
    #         peak  = torch.cuda.max_memory_allocated()  / 1024 ** 3
    #         nvml  = SonarBind._get_nvml_mem_gb()
    #         print(f'[MEM] {tag:<50s}  '
    #               f'torch_alloc={alloc:.3f}GB  torch_resv={resv:.3f}GB  '
    #               f'torch_peak={peak:.3f}GB  nvml({nvml})')
    #
    #     snap('── baseline (before data preprocess) ──')
    #
    #     # ── Stage A: data preprocessor ───────────────────────────────────────
    #     with record_function('A_data_preprocess'):
    #         with optim_wrapper.optim_context(self):
    #             data = self.data_preprocessor(data, True)
    #     snap('A. after data_preprocessor')
    #
    #     # ── Stage B: forward pass (loss() is called inside) ──────────────────
    #     with record_function('B_forward_loss'):
    #         with optim_wrapper.optim_context(self):
    #             losses = self._run_forward(data, mode='loss')
    #     parsed_losses, log_vars = self.parse_losses(losses)
    #     torch.cuda.synchronize()
    #     snap('B. after forward pass  (loss computed)')
    #
    #     # ── Stage C: backward pass ────────────────────────────────────────────
    #     with record_function('C_backward'):
    #         optim_wrapper.backward(parsed_losses)
    #     snap('C. after backward      (gradients allocated)')
    #
    #     # ── Stage D: gradient clip + optimizer step ───────────────────────────
    #     with record_function('D_optimizer_step'):
    #         optim_wrapper.step()
    #     snap('D. after optimizer step (AdamW m1/m2 updated)')
    #
    #     # ── Stage E: zero_grad ────────────────────────────────────────────────
    #     with record_function('E_zero_grad'):
    #         optim_wrapper.zero_grad()
    #     snap('E. after zero_grad     (gradients freed)')
    #
    #     torch.cuda.synchronize()
    #     peak_total = torch.cuda.max_memory_allocated() / 1024 ** 3
    #     resv_total = torch.cuda.max_memory_reserved()  / 1024 ** 3
    #     nvml_total = SonarBind._get_nvml_mem_gb()
    #     print(f'\n[MEM] ★ torch peak allocated (step) : {peak_total:.3f} GB')
    #     print(f'[MEM] ★ torch peak reserved  (step) : {resv_total:.3f} GB')
    #     print(f'[MEM] ★ nvml ({nvml_total})')
    #     print(sep + '\n')
    #
    #     return log_vars

    def predict(self, batch_inputs, batch_data_samples, rescale: bool = True):
        text_prompts = []
        enhanced_text_prompts = []
        tokens_positives = []
        for data_samples in batch_data_samples:
            # Get text prompts from data_samples or generate from class names
            if hasattr(data_samples, 'text') and data_samples.text is not None:
                # If text is a tuple/list (classes), convert to string
                if isinstance(data_samples.text, (tuple, list)):
                    clean_classes = [clean_label_name(c) for c in data_samples.text]
                    text_prompt = '. '.join(clean_classes) + '.'
                    text_prompts.append(text_prompt)
                else:
                    # Use provided text prompt string
                    text_prompts.append(data_samples.text)
            else:
                # Generate text prompt from class names in metainfo
                if hasattr(data_samples, 'metainfo') and 'classes' in data_samples.metainfo:
                    classes = data_samples.metainfo['classes']
                    # Clean class names and join them
                    clean_classes = [clean_label_name(c) for c in classes]
                    text_prompt = '. '.join(clean_classes) + '.'
                    text_prompts.append(text_prompt)
                else:
                    raise ValueError(
                        'data_samples must have either text attribute or '
                        'metainfo with classes to generate text prompts')

            if 'caption_prompt' in data_samples:
                enhanced_text_prompts.append(data_samples.caption_prompt)
            else:
                enhanced_text_prompts.append(None)
            tokens_positives.append(data_samples.get('tokens_positive', None))                                                                                                                                                           
                                                                                                                                                                                                                                        
        if 'custom_entities' in batch_data_samples[0]:                                                                                                                                                                                   
            # Assuming that the `custom_entities` flag                                                                                                                                                                                   
            # inside a batch is always the same. For single image inference                                                                                                                                                              
            custom_entities = batch_data_samples[0].custom_entities                                                                                                                                                                      
        else:                                                                                                                                                                                                                            
            custom_entities = False                                                                                                                                                                                                      
        if len(text_prompts) == 1:                                                                                                                                                                                                       
            # All the text prompts are the same,                                                                                                                                                                                         
            # so there is no need to calculate them multiple times.                                                                                                                                                                      
            _positive_maps_and_prompts = [                                                                                                                                                                                               
                self.get_tokens_positive_and_prompts(                                                                                                                                                                                    
                    text_prompts[0], custom_entities, enhanced_text_prompts[0],                                                                                                                                                          
                    tokens_positives[0])                                                                                                                                                                                                 
            ] * len(batch_inputs)                                                                                                                                                                                                        
        else:                                                                                                                                                                                                                            
            _positive_maps_and_prompts = [                                                                                                                                                                                               
                self.get_tokens_positive_and_prompts(text_prompt,                                                                                                                                                                        
                                                    custom_entities,                                                                                                                                                                    
                                                    enhanced_text_prompt,                                                                                                                                                               
                                                    tokens_positive)                                                                                                                                                                    
                for text_prompt, enhanced_text_prompt, tokens_positive in zip(                                                                                                                                                           
                    text_prompts, enhanced_text_prompts, tokens_positives)                                                                                                                                                               
            ]                                                                                                                                                                                                                            
        token_positive_maps, text_prompts, _, entities = zip(
            *_positive_maps_and_prompts)

        # image feature extraction
        visual_feats = self.extract_feat(batch_inputs)

        if isinstance(text_prompts[0], list):                                                                                                                                                                                            
            # chunked text prompts, only bs=1 is supported                                                                                                                                                                               
            assert len(batch_inputs) == 1                                                                                                                                                                                                
            count = 0                                                                                                                                                                                                                    
            results_list = []                                                                                                                                                                                                            
                                                                                                                                                                                                                                        
            entities = [[item for lst in entities[0] for item in lst]]                                                                                                                                                                   
                                                                                                                                                                                                                                        
            for b in range(len(text_prompts[0])):                                                                                                                                                                                        
                text_prompts_once = [text_prompts[0][b]]                                                                                                                                                                                 
                token_positive_maps_once = token_positive_maps[0][b]                                                                                                                                                                     
                text_dict = self.language_model(text_prompts_once)                                                                                                                                                                       
                # text feature map layer                                                                                                                                                                                                 
                if self.text_feat_map is not None:                                                                                                                                                                                       
                    text_dict['embedded'] = self.text_feat_map(                                                                                                                                                                          
                        text_dict['embedded'])                                                                                                                                                                                           
                                                                                                                                                                                                                                        
                batch_data_samples[
                    0].token_positive_map = token_positive_maps_once

                head_inputs_dict = self.forward_transformer(
                    copy.deepcopy(visual_feats),
                    text_dict,
                    batch_data_samples)
                _bbox_head_inputs = {                                                                                                                                                                                     
                    k: v for k, v in head_inputs_dict.items()                                                                                                                                                             
                    if k not in ['memory_sonar', 'memory_sonar_mask', 'memory', 'memory_mask', 'spatial_shapes']                                                                                                          
                }                                                                                                                                                                                         
                                                                                                                                                                                                                                        
                pred_instances = self.bbox_head.predict(
                    **_bbox_head_inputs,
                    rescale=rescale,
                    batch_data_samples=batch_data_samples)[0]

                if len(pred_instances) > 0:
                    # Validate labels before adding offset
                    chunk_size = len(token_positive_maps_once)
                    valid_mask = pred_instances.labels < chunk_size
                    if not valid_mask.all():
                        invalid_labels = pred_instances.labels[~valid_mask].tolist()
                        warnings.warn(
                            f'Chunk {b}: Found invalid labels {invalid_labels} (>= {chunk_size}) '
                            f'in chunk prediction. These predictions will be filtered out.')
                        # Filter out invalid predictions
                        pred_instances.bboxes = pred_instances.bboxes[valid_mask]
                        pred_instances.scores = pred_instances.scores[valid_mask]
                        pred_instances.labels = pred_instances.labels[valid_mask]

                    if len(pred_instances) > 0:  # Check again after filtering
                        pred_instances.labels += count
                count += len(token_positive_maps_once)
                results_list.append(pred_instances)                                                                                                                                                                                      
            results_list = [results_list[0].cat(results_list)]                                                                                                                                                                           
            is_rec_tasks = [False] * len(results_list)                                                                                                                                                                                   
        else:                                                                                                                                                                                                                            
            # extract text feats                                                                                                                                                                                                         
            text_dict = self.language_model(list(text_prompts))                                                                                                                                                                          
            # text feature map layer                                                                                                                                                                                                     
            if self.text_feat_map is not None:                                                                                                                                                                                           
                text_dict['embedded'] = self.text_feat_map(                                                                                                                                                                              
                    text_dict['embedded'])                                                                                                                                                                                               
                                                                                                                                                                                                                                        
            is_rec_tasks = []                                                                                                                                                                                                            
            for i, data_samples in enumerate(batch_data_samples):                                                                                                                                                                        
                if token_positive_maps[i] is not None:                                                                                                                                                                                   
                    is_rec_tasks.append(False)                                                                                                                                                                                           
                else:                                                                                                                                                                                                                    
                    is_rec_tasks.append(True)                                                                                                                                                                                            
                data_samples.token_positive_map = token_positive_maps[i]

            head_inputs_dict = self.forward_transformer(
                visual_feats,
                text_dict,
                batch_data_samples)

            _bbox_head_inputs = {                                                                                                                                                                                     
                k: v for k, v in head_inputs_dict.items()                                                                                                                                                             
                if k not in ['memory_sonar', 'memory_sonar_mask', 'memory', 'memory_mask', 'spatial_shapes']                                                                                                          
            }                                                                                                                                                                                             
                                                                                                                                                                                                                                        
            results_list = self.bbox_head.predict(
                **_bbox_head_inputs,
                rescale=rescale,
                batch_data_samples=batch_data_samples)

        # Filter out invalid labels before passing to evaluator
        num_classes = len(entities[0]) if entities and len(entities) > 0 else 0
        for i, pred_instances in enumerate(results_list):
            if len(pred_instances) > 0 and num_classes > 0:
                valid_mask = pred_instances.labels < num_classes
                if not valid_mask.all():
                    invalid_labels = pred_instances.labels[~valid_mask].tolist()
                    warnings.warn(
                        f'Found invalid labels {invalid_labels} (>= {num_classes}) '
                        f'in prediction results. These predictions will be filtered out. '
                        f'This may indicate an issue with the model prediction.')
                    # Filter out invalid predictions
                    results_list[i] = pred_instances[valid_mask]

        for data_sample, pred_instances, entity, is_rec_task in zip(                                                                                                                                                                     
                batch_data_samples, results_list, entities, is_rec_tasks):                                                                                                                                                               
            if len(pred_instances) > 0:                                                                                                                                                                                                  
                label_names = []                                                                                                                                                                                                         
                for labels in pred_instances.labels:                                                                                                                                                                                     
                    if is_rec_task:                                                                                                                                                                                                      
                        label_names.append(entity)                                                                                                                                                                                       
                        continue                                                                                                                                                                                                         
                    if labels >= len(entity):                                                                                                                                                                                            
                        warnings.warn(                                                                                                                                                                                                   
                            'The unexpected output indicates an issue with '                                                                                                                                                             
                            'named entity recognition. You can try '                                                                                                                                                                     
                            'setting custom_entities=True and running '                                                                                                                                                                  
                            'again to see if it helps.')                                                                                                                                                                                 
                        label_names.append('unobject')                                                                                                                                                                                   
                    else:                                                                                                                                                                                                                
                        label_names.append(entity[labels])                                                                                                                                                                               
                # for visualization                                                                                                                                                                                                      
                pred_instances.label_names = label_names                                                                                                                                                                                 
            data_sample.pred_instances = pred_instances                                                                                                                                                                                  
        return batch_data_samples
