import json, os
import numpy as np
from mmcv.transforms.base import BaseTransform
from mmdet.registry import TRANSFORMS

@TRANSFORMS.register_module()
class LoadSonarAnnotations(BaseTransform):
    def __init__(self, ann_file):
        d=json.load(open(ann_file)); self.by_file={x['file_name']:x['id'] for x in d['images']}; self.boxes={}
        self.label_map={c['id']:i for i,c in enumerate(sorted(d['categories'], key=lambda x:x['id']))}
        for a in d['annotations']: self.boxes.setdefault(a['image_id'], []).append(a)
    def transform(self, results):
        anns=self.boxes.get(self.by_file.get(os.path.basename(results['img_path']),-1),[])
        b=np.asarray([a['bbox'] for a in anns],dtype=np.float32)
        if len(b): b[:,2:]+=b[:,:2]
        results['sonar_gt_bboxes']=b; results['sonar_gt_labels']=np.asarray([self.label_map[a['category_id']] for a in anns],dtype=np.int64); return results
