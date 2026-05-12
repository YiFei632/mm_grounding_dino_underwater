# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import os
import os.path as osp
from datetime import datetime

import torch
from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet.utils import setup_cache_size_limit_of_dynamo


def print_model_stats(runner):
    """Print model parameter counts and GPU memory occupied by weights,
    and save the same data to a JSON file in the work_dir.

    Called after Runner.from_cfg() so pretrained weights are already loaded,
    but before runner.train() so AdamW m1/m2 have not yet been allocated.
    Only runs on rank 0 to avoid duplicated output in multi-GPU runs.

    NOTE: FlexibleRunner (used with DeepSpeedStrategy) builds the model
    lazily inside train(). In that case runner.model is still a ConfigDict
    here, so we skip stats silently and let DeepSpeed print its own summary.
    """
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if local_rank != 0:
        return

    # ── Resolve the actual nn.Module ────────────────────────────────
    import torch.nn as nn
    model = runner.model
    # Unwrap DDP / DeepSpeedEngine wrappers
    while hasattr(model, 'module'):
        model = model.module

    # FlexibleRunner with DeepSpeedStrategy keeps model as ConfigDict
    # until train() is called — bail out gracefully in that case
    if not isinstance(model, nn.Module):
        print('[model_stats] FlexibleRunner detected: model not yet '
              'instantiated before train(). Skipping pre-train stats.\n')
        return

    sep = '=' * 65

    # ── 1. GPU memory (weights only) ────────────────────────────────
    torch.cuda.synchronize()
    gpu_alloc = torch.cuda.memory_allocated() / 1024 ** 3
    gpu_resv  = torch.cuda.memory_reserved()  / 1024 ** 3

    # ── 2. Parameter counts ──────────────────────────────────────────
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters()
                           if p.requires_grad)
    frozen_params    = total_params - trainable_params

    # ── 3. Per-submodule breakdown ───────────────────────────────────
    candidate_names = [
        ('backbone',        'backbone       (RGB ResNet50)'),
        ('sonar_backbone',  'sonar_backbone (Sonar ResNet50)'),
        ('language_model',  'language_model (BERT)'),
        ('neck',            'neck           (ChannelMapper)'),
        ('encoder',         'encoder        (Transformer)'),
        ('decoder',         'decoder        (Transformer)'),
        ('bbox_head',       'bbox_head      (GroundingDINO)'),
    ]
    submodule_stats = []
    for attr, label in candidate_names:
        if hasattr(model, attr):
            mod     = getattr(model, attr)
            p_total = sum(p.numel() for p in mod.parameters())
            p_train = sum(p.numel() for p in mod.parameters()
                          if p.requires_grad)
            submodule_stats.append({
                'name':      label,
                'total':     p_total,
                'trainable': p_train,
                'frozen':    p_total - p_train,
            })

    # ── 4. Print to terminal ─────────────────────────────────────────
    print(f'\n{sep}')
    print(' Model Statistics  (rank 0, weights loaded, before training)')
    print(sep)
    print(f'  GPU memory (weights only, no activations / gradients):')
    print(f'    allocated  : {gpu_alloc:.3f} GB')
    print(f'    reserved   : {gpu_resv:.3f} GB')
    print(f'  Theoretical  : '
          f'{total_params * 4 / 1024**3:.3f} GB (fp32)  '
          f'{total_params * 2 / 1024**3:.3f} GB (fp16)')
    print()
    print(f'  {"":30s}  {"Total":>12}  {"Trainable":>12}')
    print(f'  {"-"*30}  {"-"*12}  {"-"*12}')
    print(f'  {"All parameters":<30}  '
          f'{total_params:>12,}  {trainable_params:>12,}')
    print(f'  {"  of which frozen":<30}  '
          f'{frozen_params:>12,}  {"0":>12}')
    print(f'  {"  (M)":<30}  '
          f'{total_params/1e6:>11.1f}M  {trainable_params/1e6:>11.1f}M')
    print()
    print(f'  {"Submodule":<38}  {"Params":>10}  {"Trainable":>10}')
    print(f'  {"-"*38}  {"-"*10}  {"-"*10}')
    for s in submodule_stats:
        frozen_mark = '  [frozen]' if s['trainable'] == 0 else ''
        print(f'  {s["name"]:<38}  {s["total"]:>10,}  '
              f'{s["trainable"]:>10,}{frozen_mark}')
    print(sep + '\n')

    # ── 5. Save to JSON ──────────────────────────────────────────────
    stats = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'gpu_memory': {
            'allocated_GB': round(gpu_alloc, 4),
            'reserved_GB':  round(gpu_resv,  4),
            'theoretical_fp32_GB': round(total_params * 4 / 1024**3, 4),
            'theoretical_fp16_GB': round(total_params * 2 / 1024**3, 4),
        },
        'parameters': {
            'total':     total_params,
            'trainable': trainable_params,
            'frozen':    frozen_params,
            'total_M':   round(total_params     / 1e6, 2),
            'trainable_M': round(trainable_params / 1e6, 2),
        },
        'submodules': submodule_stats,
    }

    out_dir = runner.work_dir
    os.makedirs(out_dir, exist_ok=True)
    out_path = osp.join(out_dir, 'model_stats.json')
    with open(out_path, 'w') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f'[model_stats] saved → {out_path}\n')


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    # Reduce the number of repeated compilations and improve
    # training speed.
    setup_cache_size_limit_of_dynamo()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.loss_scale = 'dynamic'

    # enable automatically scaling LR
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    # resume is determined in this priority: resume from > auto_resume
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # print model statistics before training starts
    print_model_stats(runner)

    # start training
    runner.train()


if __name__ == '__main__':
    main()
