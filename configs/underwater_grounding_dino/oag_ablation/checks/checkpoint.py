import sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import torch
from mmengine.config import Config
from mmengine.utils import import_modules_from_strings
from mmdet.registry import MODELS
from mmdet.utils import register_all_modules
register_all_modules()
torch.set_num_threads(2)
root=Path('configs/underwater_grounding_dino/oag_ablation')
# Load each architecture at its full intended depth and compare checkpoint tensors.
checkpoint=torch.load('work_dirs/oag_train/single_fusion_rgbs50/epoch_12.pth',map_location='cpu')
state=checkpoint['state_dict']
print('checkpoint tensors',len(state),flush=True)
for folder in ('00_baseline','01_no_sonar','02_no_text','03_no_fft'):
    cfg=Config.fromfile(root/folder/'rgbs50_finetune_20e.py')
    import_modules_from_strings(**cfg.custom_imports)
    model=MODELS.build(cfg.model)
    current=model.state_dict()
    bad=[k for k in current if k in state and current[k].shape != state[k].shape]
    assert not bad, bad
    result=model.load_state_dict(state,strict=False)
    if folder in ('00_baseline','03_no_fft'):
        assert not result.missing_keys and not result.unexpected_keys, result
    if folder=='01_no_sonar':
        assert not result.missing_keys, result
    if folder=='02_no_text':
        assert result.missing_keys==['category_vectors.weight'], result.missing_keys
        layer=model.decoder.layers[0]
        assert torch.equal(layer.norms[1].weight,state['decoder.layers.0.norms.2.weight'])
        assert torch.equal(layer.norms[2].weight,state['decoder.layers.0.norms.3.weight'])
        # A checkpoint from this architecture must load strictly without remapping.
        model.load_state_dict(model.state_dict(),strict=True)
    print(folder,'full model checkpoint OK; missing',result.missing_keys,
          'discarded tensors',len(result.unexpected_keys),flush=True)
    del model,current,result
    import gc
    gc.collect()
for file in sorted(root.glob('*/*.py')):
    cfg=Config.fromfile(file)
    assert cfg.resume is False
    assert cfg.train_dataloader.dataset.pipeline==cfg.train_pipeline
    assert cfg.val_dataloader.dataset.pipeline==cfg.test_pipeline
    print('CONFIG OK',file,flush=True)
