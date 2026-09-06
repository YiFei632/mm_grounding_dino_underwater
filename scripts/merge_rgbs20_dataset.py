#!/usr/bin/env python3
"""Create an RGBS20 RGB+sonar four-channel COCO dataset.

The source RGBS20 dataset stores RGB and sonar frames in separate flat
directories with matching filenames.  This script writes HWC uint8 ``.npy``
files in ``RGBS20_merged/{train,val}`` and rewrites the COCO annotations to
refer to those files.  Samples without a paired sonar frame are excluded by
default; use ``--allow-missing`` only when a grayscale-RGB fallback is
acceptable for an experiment.
"""

import argparse
import copy
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rgb-root', type=Path,
                   default=Path('/media/fishyu/fish-14tb-2/YiFei/Datasets/RGBS20-RGB'))
    p.add_argument('--sonar-root', type=Path,
                   default=Path('/media/fishyu/fish-14tb-2/YiFei/Datasets/RGBS20-Sonar'))
    p.add_argument('--output-root', type=Path,
                   default=Path('/media/fishyu/fish-14tb-2/YiFei/Datasets/RGBS20_merged'))
    p.add_argument('--allow-missing', action='store_true',
                   help='Use grayscale RGB as sonar when a pair is missing.')
    return p.parse_args()


def image_path(root: Path, file_name: str) -> Path:
    """Resolve flat RGBS20 image paths while tolerating annotation prefixes."""
    name = Path(file_name).name
    candidates = [root / 'Images' / name, root / name]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def merge_split(split: str, args) -> None:
    ann_path = args.rgb_root / f'{split}.json'
    if not ann_path.is_file():
        raise FileNotFoundError(f'Missing annotation file: {ann_path}')
    annotations = json.loads(ann_path.read_text())
    output_dir = args.output_root / split
    output_dir.mkdir(parents=True, exist_ok=True)

    kept_images = []
    kept_ids = set()
    missing = []
    for info in annotations.get('images', []):
        rgb_path = image_path(args.rgb_root, info['file_name'])
        sonar_path = image_path(args.sonar_root, info['file_name'])
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        sonar = cv2.imread(str(sonar_path), cv2.IMREAD_GRAYSCALE)
        if rgb is None or sonar is None:
            missing.append(info['file_name'])
            if not args.allow_missing:
                continue
            if rgb is None:
                raise FileNotFoundError(f'Missing RGB image: {rgb_path}')
            sonar = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

        # Store RGB in RGB order, followed by one grayscale sonar channel.
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        if sonar.shape[:2] != rgb.shape[:2]:
            sonar = cv2.resize(sonar, (rgb.shape[1], rgb.shape[0]),
                               interpolation=cv2.INTER_LINEAR)
        merged = np.dstack((rgb, sonar)).astype(np.uint8, copy=False)
        out_name = f"{Path(info['file_name']).stem}.npy"
        np.save(output_dir / out_name, merged)
        new_info = copy.deepcopy(info)
        new_info['file_name'] = out_name
        new_info['height'], new_info['width'] = merged.shape[:2]
        kept_images.append(new_info)
        kept_ids.add(info['id'])

    output_ann = copy.deepcopy(annotations)
    output_ann['images'] = kept_images
    output_ann['annotations'] = [
        ann for ann in annotations.get('annotations', [])
        if ann['image_id'] in kept_ids
    ]
    ann_out = args.output_root / 'annotations' / f'instances_{split}.json'
    ann_out.parent.mkdir(parents=True, exist_ok=True)
    ann_out.write_text(json.dumps(output_ann, indent=2) + '\n')
    print(f'{split}: kept {len(kept_images)}/{len(annotations.get("images", []))}; '
          f'annotations {len(output_ann["annotations"])}; missing pairs {len(missing)}')
    if missing:
        print('  missing:', ', '.join(map(str, missing)))


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    for split in ('train', 'val'):
        merge_split(split, args)
    print(f'Wrote merged dataset to {args.output_root}')


if __name__ == '__main__':
    main()
