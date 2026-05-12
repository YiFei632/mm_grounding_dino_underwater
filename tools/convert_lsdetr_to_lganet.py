#!/usr/bin/env python3
"""
Convert LS-DETR checkpoint to LGANet backbone checkpoint for MMDetection.

This script extracts the backbone weights (model.0 ~ model.7) from a trained
LS-DETR model and converts them to the LGANet backbone format used in MMDetection.

Usage:
    python tools/convert_lsdetr_to_lganet.py \
        --input /path/to/lsdetr_checkpoint.pt \
        --output /path/to/lganet_backbone.pth

Key mapping:
    LS-DETR (YAML indices)          LGANet (MMDetection)
    ----------------------          --------------------
    model.0 (ConvNormLayer)    ->   stem.0
    model.1 (ConvNormLayer)    ->   stem.1
    model.2 (ConvNormLayer)    ->   stem.2
    model.3 (MaxPool2d)        ->   stem.3
    model.4 (Blocks, 64ch)     ->   layer1
    model.5 (Blocks, 128ch)    ->   layer2
    model.6 (Blocks, 256ch)    ->   layer3
    model.7 (Blocks, 512ch)    ->   layer4
"""

import argparse
import torch
from collections import OrderedDict


def convert_lsdetr_to_lganet(lsdetr_ckpt_path, output_path):
    """
    Extract and convert backbone weights from LS-DETR checkpoint.

    Args:
        lsdetr_ckpt_path (str): Path to LS-DETR checkpoint (.pt file)
        output_path (str): Path to save converted LGANet backbone checkpoint
    """
    print(f"Loading LS-DETR checkpoint from: {lsdetr_ckpt_path}")
    checkpoint = torch.load(lsdetr_ckpt_path, map_location='cpu')

    # LS-DETR checkpoint structure may vary, try common keys
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        # Assume the checkpoint is already a state_dict
        state_dict = checkpoint

    print(f"Original checkpoint contains {len(state_dict)} keys")

    # Filter backbone keys (model.0 ~ model.7)
    backbone_keys = [k for k in state_dict.keys()
                     if k.startswith('model.0.') or k.startswith('model.1.') or
                        k.startswith('model.2.') or k.startswith('model.3.') or
                        k.startswith('model.4.') or k.startswith('model.5.') or
                        k.startswith('model.6.') or k.startswith('model.7.')]

    print(f"Found {len(backbone_keys)} backbone keys")

    if len(backbone_keys) == 0:
        print("\nWarning: No backbone keys found with 'model.X.' prefix.")
        print("Available key prefixes:")
        prefixes = set([k.split('.')[0] for k in state_dict.keys()])
        for prefix in sorted(prefixes):
            count = len([k for k in state_dict.keys() if k.startswith(prefix + '.')])
            print(f"  {prefix}.* : {count} keys")
        print("\nPlease check the checkpoint structure and adjust the script if needed.")
        return

    # Create mapping
    new_state_dict = OrderedDict()

    for old_key in backbone_keys:
        # Determine new key prefix
        if old_key.startswith('model.0.'):
            new_key = old_key.replace('model.0.', 'stem.0.')
        elif old_key.startswith('model.1.'):
            new_key = old_key.replace('model.1.', 'stem.1.')
        elif old_key.startswith('model.2.'):
            new_key = old_key.replace('model.2.', 'stem.2.')
        elif old_key.startswith('model.3.'):
            new_key = old_key.replace('model.3.', 'stem.3.')
        elif old_key.startswith('model.4.'):
            new_key = old_key.replace('model.4.', 'layer1.')
        elif old_key.startswith('model.5.'):
            new_key = old_key.replace('model.5.', 'layer2.')
        elif old_key.startswith('model.6.'):
            new_key = old_key.replace('model.6.', 'layer3.')
        elif old_key.startswith('model.7.'):
            new_key = old_key.replace('model.7.', 'layer4.')
        else:
            continue

        new_state_dict[new_key] = state_dict[old_key]

    print(f"Converted {len(new_state_dict)} parameters")

    # Print sample keys for verification
    print("\nSample converted keys:")
    for i, (old_key, new_key) in enumerate(zip(backbone_keys[:5], new_state_dict.keys())):
        print(f"  {old_key}")
        print(f"    -> {new_key}")
        if i >= 4:
            break

    # Save converted checkpoint
    # MMDetection expects 'state_dict' key
    output_checkpoint = {'state_dict': new_state_dict}

    torch.save(output_checkpoint, output_path)
    print(f"\nSaved LGANet backbone checkpoint to: {output_path}")
    print(f"Total parameters: {len(new_state_dict)}")

    # Print layer statistics
    print("\nLayer statistics:")
    for layer_name in ['stem.0', 'stem.1', 'stem.2', 'stem.3',
                       'layer1', 'layer2', 'layer3', 'layer4']:
        count = len([k for k in new_state_dict.keys() if k.startswith(layer_name + '.')])
        if count > 0:
            print(f"  {layer_name}: {count} parameters")


def main():
    parser = argparse.ArgumentParser(
        description='Convert LS-DETR checkpoint to LGANet backbone checkpoint')
    parser.add_argument('--input', '-i', required=True,
                        help='Path to LS-DETR checkpoint (.pt file)')
    parser.add_argument('--output', '-o', required=True,
                        help='Path to save LGANet backbone checkpoint (.pth file)')

    args = parser.parse_args()

    convert_lsdetr_to_lganet(args.input, args.output)

    print("\n" + "="*60)
    print("Conversion complete!")
    print("="*60)
    print("\nTo use this checkpoint in your config, add:")
    print("    sonar_backbone=dict(")
    print("        type='LGANet',")
    print("        out_indices=(3,),")
    print("        frozen_stages=-1,")
    print("        norm_eval=False,")
    print(f"        init_cfg=dict(type='Pretrained', checkpoint='{args.output}')")
    print("    ),")


if __name__ == '__main__':
    main()
