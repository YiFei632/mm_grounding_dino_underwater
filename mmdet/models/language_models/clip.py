# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from typing import Sequence

import torch
from mmengine.model import BaseModel
from torch import nn

try:                                                                                                                                                                                                         
    from transformers import CLIPTokenizerFast, CLIPTextConfig                                                                                                                                               
    from transformers import CLIPTextModel as HFCLIPTextModel                                                                                                                                                
except ImportError:                                                                                                                                                                                          
    CLIPTokenizerFast = None                                                                                                                                                                                 
    HFCLIPTextModel = None                                                                                                                                                                                   
    CLIPTextConfig = None

from mmdet.registry import MODELS


def generate_masks_with_special_tokens_and_transfer_map(
        tokenized, special_tokens_list):
    """Generate attention mask between each pair of special tokens.

    Only token pairs in between two special tokens are attended to
    and thus the attention mask for these pairs is positive.

    Args:
        input_ids (torch.Tensor): input ids. Shape: [bs, num_token]
        special_tokens_list (list): special tokens list.

    Returns:
        Tuple(Tensor, Tensor):
        - attention_mask is the attention mask between each tokens.
          Only token pairs in between two special tokens are positive.
          Shape: [bs, num_token, num_token].
        - position_ids is the position id of tokens within each valid sentence.
          The id starts from 0 whenenver a special token is encountered.
          Shape: [bs, num_token]
    """
    input_ids = tokenized['input_ids']
    bs, num_token = input_ids.shape
    # special_tokens_mask:
    # bs, num_token. 1 for special tokens. 0 for normal tokens
    special_tokens_mask = torch.zeros((bs, num_token),
                                      device=input_ids.device).bool()

    for special_token in special_tokens_list:
        special_tokens_mask |= input_ids == special_token

    # idxs: each row is a list of indices of special tokens
    idxs = torch.nonzero(special_tokens_mask)

    # generate attention mask and positional ids
    attention_mask = (
        torch.eye(num_token,
                  device=input_ids.device).bool().unsqueeze(0).repeat(
                      bs, 1, 1))
    position_ids = torch.zeros((bs, num_token), device=input_ids.device)
    previous_col = 0
    for i in range(idxs.shape[0]):
        row, col = idxs[i]
        if (col == 0) or (col == num_token - 1):
            attention_mask[row, col, col] = True
            position_ids[row, col] = 0
        else:
            attention_mask[row, previous_col + 1:col + 1,
                           previous_col + 1:col + 1] = True
            position_ids[row, previous_col + 1:col + 1] = torch.arange(
                0, col - previous_col, device=input_ids.device)
        previous_col = col

    return attention_mask, position_ids.to(torch.long)


@MODELS.register_module()
class CLIPModel(BaseModel):
    """CLIP language model for text embedding.

    Args:
        name (str, optional): name of the pretrained CLIP model from
            HuggingFace. Defaults to 'openai/clip-vit-base-patch32'.
        max_tokens (int, optional): maximum number of tokens to be
            used for CLIP. CLIP supports up to 77 tokens. Defaults to 77.
        pad_to_max (bool, optional): whether to pad the tokens to max_tokens.
             Defaults to True.
        use_sub_sentence_represent (bool, optional): whether to use sub
            sentence represent introduced in `Grounding DINO
            <https://arxiv.org/abs/2303.05499>`. Defaults to False.
        special_tokens_list (list, optional): special tokens used to split
            subsentence. It cannot be None when `use_sub_sentence_represent`
            is True. Defaults to None.
        use_checkpoint (bool, optional): whether to use gradient checkpointing.
             Defaults to False.
    """

    def __init__(self,
                 name: str = 'clip-vit-base-patch32',
                 max_tokens: int = 77,
                 pad_to_max: bool = True,
                 use_sub_sentence_represent: bool = False,
                 special_tokens_list: list = None,
                 use_checkpoint: bool = False,
                 **kwargs) -> None:

        super().__init__(**kwargs)
        self.max_tokens = max_tokens
        self.pad_to_max = pad_to_max

        if CLIPTokenizerFast is None:
            raise RuntimeError(
                'transformers is not installed, please install it by: '
                'pip install transformers.')

        self.tokenizer = CLIPTokenizerFast.from_pretrained(name)
        self.language_backbone = nn.Sequential(
            OrderedDict([('body',
                          CLIPEncoder(
                              name,
                              use_checkpoint=use_checkpoint))]))

        self.use_sub_sentence_represent = use_sub_sentence_represent
        if self.use_sub_sentence_represent:
            assert special_tokens_list is not None, \
                'special_tokens should not be None \
                    if use_sub_sentence_represent is True'

            self.special_tokens = self.tokenizer.convert_tokens_to_ids(
                special_tokens_list)

    def forward(self, captions: Sequence[str], **kwargs) -> dict:
        """Forward function."""
        device = next(self.language_backbone.parameters()).device
        tokenized = self.tokenizer(
            captions,
            max_length=self.max_tokens,
            padding='max_length' if self.pad_to_max else 'longest',
            return_tensors='pt',
            truncation=True)

        # Move to device
        tokenized = {k: v.to(device) for k, v in tokenized.items()}

        input_ids = tokenized['input_ids']
        if self.use_sub_sentence_represent:
            attention_mask, position_ids = \
                generate_masks_with_special_tokens_and_transfer_map(
                    tokenized, self.special_tokens)
        else:
            attention_mask = tokenized['attention_mask']
            position_ids = None

        tokenizer_input = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids
        }
        language_dict_features = self.language_backbone(tokenizer_input)
        if self.use_sub_sentence_represent:
            language_dict_features['position_ids'] = position_ids
            language_dict_features[
                'text_token_mask'] = tokenized['attention_mask'].bool()
        return language_dict_features


class CLIPEncoder(nn.Module):
    """CLIP text encoder for language embedding.

    Args:
        name (str): name of the pretrained CLIP model from HuggingFace.
                Defaults to 'openai/clip-vit-base-patch32'.
        use_checkpoint (bool): whether to use gradient checkpointing.
                Defaults to False.
    """

    def __init__(self,
                 name: str,
                 use_checkpoint: bool = False):
        super().__init__()
        if CLIPTextConfig is None:
            raise RuntimeError(
                'transformers is not installed, please install it by: '
                'pip install transformers.')
        config = CLIPTextConfig.from_pretrained(name)
        config.gradient_checkpointing = use_checkpoint

        # Load CLIP text model
        self.model = HFCLIPTextModel.from_pretrained(name, config=config)
        self.language_dim = config.hidden_size

    def forward(self, x) -> dict:
        """Forward function.

        Args:
            x (dict): Input dictionary containing:
                - input_ids: token ids
                - attention_mask: attention mask
                - position_ids: position ids (optional)

        Returns:
            dict: Dictionary containing:
                - embedded: text embeddings
                - masks: attention mask
                - hidden: last hidden state
        """
        input_ids = x['input_ids']
        attention_mask = x['attention_mask']
        position_ids = x.get('position_ids', None)

        # CLIP text model forward
        if attention_mask.dim() == 3:                                                                                                                                                                                
        # 使用简单的2维mask或者不传mask                                                                                                                                                                          
            clip_attention_mask = None  # 让CLIP使用默认mask                                                                                                                                                         
        else:                                                                                                                                                                                                        
            clip_attention_mask = attention_mask                                                                                                                                                                     
                                                                                                                                                                                                                    
        outputs = self.model(                                                                                                                                                                                        
            input_ids=input_ids,                                                                                                                                                                                     
            attention_mask=clip_attention_mask,  # 使用处理后的mask                                                                                                                                                  
            position_ids=position_ids,                                                                                                                                                                               
            output_hidden_states=True,                                                                                                                                                                               
            return_dict=True                                                                                                                                                                                         
        )

        # Get the last hidden state
        # CLIP outputs: last_hidden_state, pooler_output, hidden_states
        hidden_states = outputs.hidden_states

        # Use the last hidden state as features
        features = hidden_states[-1]

        # Apply attention mask
        if attention_mask.dim() == 2:
            embedded = features * attention_mask.unsqueeze(-1).float()
        else:
            embedded = features

        results = {
            'embedded': embedded,
            'masks': attention_mask,
            'hidden': features
        }
        return results
