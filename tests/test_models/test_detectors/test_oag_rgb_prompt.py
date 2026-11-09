"""Regression checks for the opt-in RGB-coordinate sonar prompt path."""
import gc
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample
from mmdet.registry import MODELS
from mmdet.utils import register_all_modules
from mmdet.models.detectors.oag_rgb_prompt import OAGRGBPrompt
from mmdet.models.utils.sonar_rgb_prompt import (
    SonarRGBPrompt, prompt_token_bias, box_class_targets, spatial_bce_dice)
from mmdet.models.layers.transformer.oag_rgb_prompt_decoder import RGBPromptAttention


ROOT = Path(__file__).resolve().parents[3]
CONFIGS = ROOT / 'configs/underwater_grounding_dino/oag'


def test_transport_padding_null_and_gradients():
    torch.manual_seed(7)
    mapper = SonarRGBPrompt(16)
    rgb = torch.randn(2, 6, 16, requires_grad=True)
    sonar = torch.randn(2, 16, 2, 2, requires_grad=True)
    text = torch.randn(2, 3, 16, requires_grad=True)
    valid = torch.tensor([[True, True, False], [True, True, True]])
    rgb_mask = torch.tensor([[False] * 5 + [True], [False] * 6])
    sonar_mask = torch.tensor([[False, False, True, True], [True] * 4])
    prompt, logits, _, null = mapper(rgb, sonar, text, valid, rgb_mask, sonar_mask)
    assert torch.isfinite(prompt).all()
    assert (prompt[0, -1] == 0).all()
    assert (prompt[0, :, 2] == 0).all()
    assert (prompt[1] == 0).all()
    assert (null[1] == 1).all()
    assert (prompt >= 0).all() and (prompt <= 1).all()
    loss = spatial_bce_dice(prompt, torch.zeros_like(prompt),
                            (~rgb_mask)[..., None] & valid[:, None])
    loss = loss + spatial_bce_dice(logits, torch.zeros_like(logits),
                                  (~sonar_mask)[..., None] & valid[:, None], logits=True)
    loss.backward()
    for tensor in (rgb, sonar, text):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
    assert mapper.rgb_match.weight.grad.abs().sum() > 0
    assert mapper.null_logit.grad.abs() > 0


def test_selector_bias_changes_ranking_without_special_tokens():
    prompt = torch.tensor([[[0., 0.], [0., 1.]]])
    token_map = torch.tensor([[[0., 1., 0., 0.], [0., 0., 0.5, 0.5]]])
    bias = prompt_token_bias(prompt, token_map, 6)
    assert bias[0, 1].tolist() == [0., 0., 1., 1., 0., 0.]
    logits = torch.tensor([[[0., 0.3, 0.1, 0.1, -10., -10.],
                            [0., 0.1, 0.2, 0.2, -10., -10.]]])
    assert logits.max(-1).values.argmax(-1).item() == 0
    assert (logits + 0.5 * bias).max(-1).values.argmax(-1).item() == 1


def test_decoder_zero_prompt_identity_and_nonzero_gradients():
    torch.manual_seed(2)
    module = RGBPromptAttention(16, 4)
    query = torch.randn(2, 3, 16)
    pos = torch.randn(2, 4, 16)
    classes = torch.randn(2, 2, 16)
    valid = torch.ones(2, 2, dtype=torch.bool)
    mask = torch.tensor([[False] * 4, [True] * 4])
    zero = torch.zeros(2, 4, 2)
    output = module(query, None, pos, zero, classes, valid, mask)
    assert torch.equal(output, query)
    prior = torch.rand(2, 4, 2, requires_grad=True)
    output = module(query, None, pos, prior, classes, valid, mask)
    assert torch.equal(output[1], query[1])
    assert not torch.equal(output[0], query[0])
    output.square().mean().backward()
    assert prior.grad[0].abs().sum() > 0
    module.weight = 0
    assert torch.equal(module(query, None, pos, prior, classes, valid, mask), query)


def test_box_targets_use_padded_rgb_canvas():
    instances = InstanceData(bboxes=torch.tensor([[32., 0., 64., 32.]]),
                             labels=torch.tensor([1]))
    target = box_class_targets(instances, (64, 128), (2, 4), 2, 'cpu')
    assert target[:, 1].tolist() == [0., 1., 0., 0., 0., 0., 0., 0.]


def make_sample(empty_sonar=False):
    sample = DetDataSample(metainfo=dict(
        img_shape=(128, 128), ori_shape=(128, 128), batch_input_shape=(128, 128),
        scale_factor=(1., 1.), sonar_shape=(64, 64),
        text=('iron_ball', 'uuv'), custom_entities=True))
    sample.gt_instances = InstanceData(bboxes=torch.tensor([[20., 20., 70., 80.]]),
                                      labels=torch.tensor([0]))
    sample.sonar_img = torch.rand(1, 64, 64)
    sample.sonar_gt_instances = InstanceData(
        bboxes=torch.empty(0, 4) if empty_sonar else torch.tensor([[4., 5., 40., 50.]]),
        labels=torch.empty(0, dtype=torch.long) if empty_sonar else torch.tensor([0]))
    return sample


def _check_model_train_predict_cpu(name):
    """Real backbones/BERT/heads, reduced spatial and transformer sizes for CPU."""
    register_all_modules()
    torch.set_num_threads(2)
    cfg = Config.fromfile(str(CONFIGS / name))
    cfg.model.num_queries = 10
    cfg.model.sonar_input_size = (64, 64)
    cfg.model.encoder.num_layers = 1
    cfg.model.encoder.num_cp = 0
    cfg.model.decoder.num_layers = 1
    cfg.model.dn_cfg.group_cfg.num_dn_queries = 10
    cfg.model.backbone.init_cfg = None
    cfg.model.sonar_backbone.init_cfg = None
    model = MODELS.build(cfg.model)
    # Initialize the uninitialized level embedding without loading checkpoints.
    torch.nn.init.normal_(model.level_embed)
    model.train()
    sample = make_sample()
    inputs = torch.randn(1, 3, 128, 128)
    losses = model.loss(inputs, [sample])
    assert all(torch.isfinite(v).all() for v in losses.values())
    sum(losses.values()).backward()
    if isinstance(model, OAGRGBPrompt):
        grad = model.rgb_prompt_mapper.rgb_match.weight.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
        assert model.decoder.layers[0].rgb_prompt_attn.attn.in_proj_weight.grad is not None
        model.zero_grad(set_to_none=True)
        losses = model.loss(inputs, [make_sample(empty_sonar=True)])
        assert all(torch.isfinite(v).all() for v in losses.values())
        sum(losses.values()).backward()
    model.eval()
    with torch.no_grad():
        results = model.predict(inputs, [make_sample()])
    assert len(results) == 1
    assert torch.isfinite(results[0].pred_instances.bboxes).all()
    del model, losses
    gc.collect()


def test_original_model_train_predict_cpu():
    _check_model_train_predict_cpu('oag_double_resnet50_rgbs50_position_only_decoder.py')


def test_rgb_prompt_model_train_predict_cpu():
    _check_model_train_predict_cpu('oag_double_resnet50_rgbs50_rgb_prompt.py')


if __name__ == '__main__':
    for name, function in list(globals().items()):
        if name.startswith('test_'):
            function()
            print('PASS', name, flush=True)
