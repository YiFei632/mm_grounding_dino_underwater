import mmcv
import numpy as np
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS

@TRANSFORMS.register_module()
class LoadUAMFDetImages(BaseTransform):
    """Load paired optical and sonar files into one 6-channel image."""
    def __init__(self, to_float32=False, sonar_key='sonar_path'):
        self.to_float32, self.sonar_key = to_float32, sonar_key
    def transform(self, results):
        optical = mmcv.imread(results['img_path'], flag='color')
        sonar = mmcv.imread(results[self.sonar_key], flag='color')
        if optical.shape[:2] != sonar.shape[:2]:
            sonar = mmcv.imresize(sonar, (optical.shape[1], optical.shape[0]))
        img = np.concatenate([optical, sonar], axis=2)
        if self.to_float32: img = img.astype(np.float32)
        results['img'] = img; results['img_shape'] = img.shape[:2]; results['ori_shape'] = img.shape[:2]
        return results
