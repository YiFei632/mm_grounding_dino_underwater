"""RGBS50 -> RGBS20 transfer with new closed-set categories, without text."""
import warnings

from mmdet.registry import MODELS
from .no_text_adapter import OAGNoTextAblation


@MODELS.register_module()
class OAGNoTextRGBS20Ablation(OAGNoTextAblation):
    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        key = prefix + 'category_vectors.weight'
        if key in state_dict and state_dict[key].shape != self.category_vectors.weight.shape:
            # The two datasets have distinct class meanings, not a subset map.
            # Preserve the randomly initialized RGBS20 vectors, not source rows.
            state_dict.pop(key)
            warnings.warn('RGBS50 category vectors discarded: RGBS20 uses four '
                          'new randomly initialized category vectors.')
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
