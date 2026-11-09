import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import torch
from mmengine.config import Config
from mmdet.utils import register_all_modules
from mmdet.registry import MODELS
from mmengine.utils import import_modules_from_strings
register_all_modules()
torch.set_num_threads(2)
root = Path('configs/underwater_grounding_dino/oag_ablation')
for name in ('00_baseline', '01_no_sonar', '02_no_text', '03_no_fft'):
    cfg = Config.fromfile(root/name/'rgbs50_finetune_20e.py')
    import_modules_from_strings(**cfg.custom_imports)
    print(name, cfg.model.type, cfg.train_cfg.max_epochs, cfg.optim_wrapper.optimizer.lr, flush=True)
    cfg.model.num_queries = 10
    cfg.model.encoder.num_layers = 1
    cfg.model.encoder.num_cp = 0
    cfg.model.decoder.num_layers = 1
    cfg.model.dn_cfg.group_cfg.num_dn_queries = 10
    cfg.model.backbone.init_cfg = None
    if 'sonar_backbone' in cfg.model:
        cfg.model.sonar_backbone.init_cfg = None
        cfg.model.sonar_input_size = (64, 64)
    model = MODELS.build(cfg.model)
    torch.nn.init.normal_(model.level_embed)
    from mmengine.structures import InstanceData
    from mmdet.structures import DetDataSample
    def sample(empty=False):
        s = DetDataSample(metainfo=dict(img_shape=(128,128), ori_shape=(128,128),
            batch_input_shape=(128,128), scale_factor=(1.,1.), sonar_shape=(64,64),
            text=cfg.metainfo.classes, custom_entities=True))
        s.gt_instances = InstanceData(bboxes=torch.tensor([[20.,20.,70.,80.]]), labels=torch.tensor([0]))
        s.sonar_img = torch.rand(1,64,64)
        s.sonar_gt_instances = InstanceData(bboxes=torch.empty(0,4) if empty else torch.tensor([[4.,5.,40.,50.]]),
            labels=torch.empty(0,dtype=torch.long) if empty else torch.tensor([0]))
        return s
    x = torch.randn(1,3,128,128)
    model.train()
    losses = model.loss(x, [sample()])
    assert all(torch.isfinite(v).all() for v in losses.values()), losses
    sum(losses.values()).backward()
    if name == '01_no_sonar':
        assert not any('sonar' in n or 'rgb_prompt' in n for n, _ in model.named_parameters())
        assert not any(s['type'].startswith('LoadSonar') for s in cfg.train_dataloader.dataset.pipeline)
    if name == '02_no_text':
        assert model.language_model is None and model.text_feat_map is None
        assert not hasattr(model.encoder, 'text_layers')
        assert all(layer.cross_attn_text is None for layer in model.decoder.layers)
        grad = model.category_vectors.weight.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
        assert model.rgb_prompt_mapper.rgb_match.weight.grad.abs().sum() > 0
        model.zero_grad(set_to_none=True)
        empty_losses = model.loss(x, [sample(empty=True)])
        assert all(torch.isfinite(v).all() for v in empty_losses.values())
        sum(empty_losses.values()).backward()
    if name == '03_no_fft':
        z = torch.randn(1,2,7,9)
        assert torch.equal(model.fft_band0_filter(z), z)
    model.eval()
    with torch.no_grad():
        results = model.predict(x, [sample()])
    assert torch.isfinite(results[0].pred_instances.bboxes).all()
    assert (results[0].pred_instances.labels < 6).all()
    print('PASS loss/backward/predict', name, flush=True)
    del model, losses
    import gc
    gc.collect()
