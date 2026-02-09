#!/usr/bin/env python3
"""Convert Objects365 DSDL format to COCO format."""

import json
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm


def load_categories_from_yaml(yaml_path):
    """Load category definitions from DSDL class-domain.yaml file."""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    classes = data['Object365ClassDomain']['classes']

    # Create categories list in COCO format
    categories = []
    for idx, class_name in enumerate(classes, start=1):
        # Extract the actual class name (remove the prefix)
        name = class_name.split('.')[-1] if '.' in class_name else class_name
        categories.append({
            'id': idx,
            'name': name,
            'supercategory': class_name.split('.')[0] if '.' in class_name else 'object'
        })

    return categories


def convert_dsdl_to_coco(dsdl_json_path, class_domain_yaml_path, output_path, img_prefix=''):
    """Convert DSDL format JSON to COCO format.

    Args:
        dsdl_json_path: Path to DSDL samples JSON file
        class_domain_yaml_path: Path to class-domain.yaml file
        output_path: Path to save the converted COCO format JSON
        img_prefix: Prefix to add to image paths (e.g., 'train/' or 'val/')
    """
    print(f"Loading DSDL data from {dsdl_json_path}...")
    with open(dsdl_json_path, 'r') as f:
        dsdl_data = json.load(f)

    print(f"Loading categories from {class_domain_yaml_path}...")
    categories = load_categories_from_yaml(class_domain_yaml_path)

    # Initialize COCO format structure
    coco_data = {
        'images': [],
        'annotations': [],
        'categories': categories
    }

    print("Converting samples to COCO format...")
    annotation_id = 1  # COCO annotation IDs should start from 1

    for sample in tqdm(dsdl_data['samples']):
        media = sample['media']

        # Add image info
        image_info = {
            'id': media['id'],
            'file_name': media['media_path'],
            'height': media['media_shape'][0],
            'width': media['media_shape'][1]
        }

        # Add license if available
        if 'license' in media:
            image_info['license'] = media['license']

        coco_data['images'].append(image_info)

        # Add annotations
        for ann in sample.get('annotations', []):
            annotation = {
                'id': annotation_id,
                'image_id': media['id'],
                'category_id': ann['category_id'],
                'bbox': ann['bbox'],  # [x, y, width, height]
                'area': ann.get('area', ann['bbox'][2] * ann['bbox'][3]),
                'iscrowd': ann.get('iscrowd', 0)
            }
            coco_data['annotations'].append(annotation)
            annotation_id += 1

    # Save to output file
    print(f"Saving COCO format data to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(coco_data, f)

    print(f"Conversion complete!")
    print(f"  - Images: {len(coco_data['images'])}")
    print(f"  - Annotations: {len(coco_data['annotations'])}")
    print(f"  - Categories: {len(coco_data['categories'])}")


def main():
    parser = argparse.ArgumentParser(description='Convert Objects365 DSDL format to COCO format')
    parser.add_argument('--dsdl-root', type=str, required=True,
                        help='Root directory of DSDL dataset')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for COCO format annotations')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test'],
                        default='train', help='Dataset split to convert')

    args = parser.parse_args()

    dsdl_root = Path(args.dsdl_root)
    output_dir = Path(args.output_dir)

    # Paths
    class_domain_yaml = dsdl_root / 'defs' / 'class-domain.yaml'
    dsdl_json = dsdl_root / f'set-{args.split}' / f'{args.split}_samples.json'
    output_json = output_dir / f'{args.split}_coco.json'

    # Check if files exist
    if not class_domain_yaml.exists():
        raise FileNotFoundError(f"Class domain file not found: {class_domain_yaml}")
    if not dsdl_json.exists():
        raise FileNotFoundError(f"DSDL samples file not found: {dsdl_json}")

    # Convert
    convert_dsdl_to_coco(dsdl_json, class_domain_yaml, output_json)


if __name__ == '__main__':
    main()
