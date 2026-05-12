#!/usr/bin/env python3
"""
Extract LGANet backbone weights from trained SonarBindLGANet checkpoint.

This script extracts the sonar_backbone (LGANet) weights from a trained
SonarBindLGANet detector checkpoint and saves them in a format that can be
used to initialize LGANet in other models.

Usage:
    python tools/extract_lganet_from_sonarbind.py \
        --input work_dirs/lganet_backbone_train/epoch_50.pth \
        --output checkpoints/lganet_rgbs50_backbone.pth

The extracted checkpoint can then be used in configs:
    sonar_backbone=dict(
        type='LGANet',
        out_indices=(3,),
        frozen_stages=-1,
        norm_eval=False,
        init_cfg=dict(type='Pretrained', checkpoint='checkpoints/lganet_rgbs50_backbone.pth')
    ),
"""

import argparse
import torch
from collections import OrderedDict


def extract_lganet_backbone(checkpoint_path, output_path):
    """
    Extract sonar_backbone (LGANet) weights from SonarBindLGANet checkpoint.

    Args:
        checkpoint_path (str): Path to trained SonarBindLGANet checkpoint
        output_path (str): Path to save extracted LGANet backbone weights
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    print(f"Original checkpoint contains {len(state_dict)} keys")

    # Extract sonar_backbone keys
    backbone_state_dict = OrderedDict()
    sonar_backbone_keys = []

    for key, value in state_dict.items():
        if key.startswith('sonar_backbone.'):
            # Remove 'sonar_backbone.' prefix
            new_key = key.replace('sonar_backbone.', '')
            backbone_state_dict[new_key] = value
            sonar_backbone_keys.append(key)

    if len(backbone_state_dict) == 0:
        print("\nWarning: No sonar_backbone keys found!")
        print("Available key prefixes:")
        prefixes = set([k.split('.')[0] for k in state_dict.keys()])
        for prefix in sorted(prefixes):
            count = len([k for k in state_dict.keys() if k.startswith(prefix + '.')])
            print(f"  {prefix}.* : {count} keys")
        return

    print(f"Extracted {len(backbone_state_dict)} backbone parameters")

    # Print sample keys for verification
    print("\nSample extracted keys (original -> new):")
    for i, old_key in enumerate(sonar_backbone_keys[:10]):
        new_key = old_key.replace('sonar_backbone.', '')
        print(f"  {old_key}")
        print(f"    -> {new_key}")
        if i >= 4:
            break

    # Save in MMDetection checkpoint format
    output_checkpoint = {
        'state_dict': backbone_state_dict,
        'meta': {
            'source': 'SonarBindLGANet',
            'original_checkpoint': checkpoint_path,
            'architecture': 'LGANet',
        }
    }

    torch.save(output_checkpoint, output_path)
    print(f"\nSaved LGANet backbone checkpoint to: {output_path}")

    # Print layer statistics
    print("\nLayer statistics:")
    layer_stats = {}
    for layer_name in ['stem', 'layer1', 'layer2', 'layer3', 'layer4']:
        count = len([k for k in backbone_state_dict.keys() if k.startswith(layer_name + '.')])
        if count > 0:
            layer_stats[layer_name] = count
            print(f"  {layer_name}: {count} parameters")

    # Verify structure
    print("\nBackbone structure verification:")
    has_stem = any(k.startswith('stem.') for k in backbone_state_dict.keys())
    has_layers = all(any(k.startswith(f'layer{i}.') for k in backbone_state_dict.keys())
                     for i in range(1, 5))

    if has_stem and has_layers:
        print("  ✓ Backbone structure looks correct (stem + layer1-4)")
    else:
        print("  ✗ Warning: Backbone structure may be incomplete")
        print(f"    Has stem: {has_stem}")
        print(f"    Has layer1-4: {has_layers}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract LGANet backbone from SonarBindLGANet checkpoint')
    parser.add_argument('--input', '-i', required=True,
                        help='Path to SonarBindLGANet checkpoint (.pth file)')
    parser.add_argument('--output', '-o', required=True,
                        help='Path to save LGANet backbone checkpoint (.pth file)')

    args = parser.parse_args()

    extract_lganet_backbone(args.input, args.output)

    print("\n" + "="*70)
    print("Extraction complete!")
    print("="*70)
    print("\nTo use this checkpoint in your config, update sonar_backbone:")
    print("    sonar_backbone=dict(")
    print("        type='LGANet',")
    print("        out_indices=(3,),")
    print("        frozen_stages=-1,")
    print("        norm_eval=False,")
    print(f"        init_cfg=dict(type='Pretrained', checkpoint='{args.output}')")
    print("    ),")
    print("    sonar_feat_channels=512,")


if __name__ == '__main__':
    main()
