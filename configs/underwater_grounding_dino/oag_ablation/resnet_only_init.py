"""Keep BERT/tokenizer interfaces, discard all pretrained BERT parameters."""
from mmdet.models.language_models.bert import BertModel
from mmdet.registry import MODELS


@MODELS.register_module()
class RandomInitAblationBert(BertModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Existing wrapper loads BERT internally. Explicitly reinitialize every
        # layer, including embeddings; no pretrained BERT values are retained.
        bert = self.language_backbone.body.model
        bert.apply(bert._init_weights)
