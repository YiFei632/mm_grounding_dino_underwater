import os.path as osp
from mmdet.registry import DATASETS
from .coco import CocoDataset

@DATASETS.register_module()
class UAOFDataset(CocoDataset):
    """UAOF COCO dataset; each image entry may contain ``sonar_file``."""
    METAINFO = {'classes': ('person','mine','spherical_mine','tire','rov','auv','circle_cage','square_cage','cone','hexagonal')}
    def parse_data_info(self, raw_data_info):
        data = super().parse_data_info(raw_data_info)
        info = raw_data_info['raw_img_info']
        sonar = info.get('sonar_file', info.get('sonar_filename'))
        if sonar is None:
            raise KeyError('UAOF annotation image must contain sonar_file or sonar_filename')
        data['sonar_path'] = osp.join(self.data_prefix.get('sonar', self.data_prefix.get('img', '')), sonar)
        return data
