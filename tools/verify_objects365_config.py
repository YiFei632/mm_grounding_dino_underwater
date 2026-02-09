#!/usr/bin/env python3
"""验证 Objects365 DQDINO 训练配置是否正确"""

import json
from pathlib import Path


def verify_config():
    """验证配置是否正确"""
    print("=" * 80)
    print("Objects365 DQDINO 配置验证")
    print("=" * 80)

    # 1. 检查标注文件
    print("\n1. 检查标注文件...")
    data_root = Path("/media/fishyu/fish-14tb-2/YiFei/Dataset/Objects365")
    train_ann = data_root / "annotations/train_coco.json"
    val_ann = data_root / "annotations/val_coco.json"

    if not train_ann.exists():
        print(f"  ❌ 训练集标注文件不存在: {train_ann}")
        return False
    else:
        print(f"  ✓ 训练集标注文件存在: {train_ann}")

    if not val_ann.exists():
        print(f"  ❌ 验证集标注文件不存在: {val_ann}")
        return False
    else:
        print(f"  ✓ 验证集标注文件存在: {val_ann}")

    # 2. 检查类别数
    print("\n2. 检查标注文件中的类别数...")
    with open(val_ann) as f:
        val_data = json.load(f)

    num_categories = len(val_data['categories'])
    print(f"  标注文件中的类别数: {num_categories}")

    if num_categories != 365:
        print(f"  ❌ 类别数不正确，期望 365，实际 {num_categories}")
        return False
    else:
        print(f"  ✓ 类别数正确: 365")

    # 3. 检查图片路径
    print("\n3. 检查图片路径格式...")
    sample_train_img = json.load(open(train_ann))['images'][0]
    sample_val_img = val_data['images'][0]

    print(f"  训练集示例路径: {sample_train_img['file_name']}")
    print(f"  验证集示例路径: {sample_val_img['file_name']}")

    # 验证实际文件是否存在
    train_img_path = data_root / "raw/Objects365/data" / sample_train_img['file_name']
    val_img_path = data_root / "raw/Objects365/data" / sample_val_img['file_name']

    if not train_img_path.exists():
        print(f"  ❌ 训练集示例图片不存在: {train_img_path}")
        return False
    else:
        print(f"  ✓ 训练集示例图片存在")

    if not val_img_path.exists():
        print(f"  ❌ 验证集示例图片不存在: {val_img_path}")
        return False
    else:
        print(f"  ✓ 验证集示例图片存在")

    # 4. 检查配置文件中的类别数
    print("\n4. 检查配置文件中的类别数...")
    config_file = Path("/media/fishyu/fish-14tb-2/YiFei/Grounding_DINO/mmdetection/configs/underwater_grounding_dino/dq_dino/dqdino_pretrained_swin-l_5scale_o365.py")

    if not config_file.exists():
        print(f"  ❌ 配置文件不存在: {config_file}")
        return False

    with open(config_file) as f:
        config_content = f.read()

    # 检查 num_classes
    if "num_classes=365" in config_content:
        print(f"  ✓ bbox_head.num_classes = 365")
    else:
        print(f"  ❌ bbox_head.num_classes 不等于 365")
        return False

    # 检查 cls_num
    if "cls_num=365" in config_content:
        print(f"  ✓ ccm_cfg.cls_num = 365")
    else:
        print(f"  ❌ ccm_cfg.cls_num 不等于 365")
        return False

    # 5. 统计信息
    print("\n5. 数据集统计信息...")
    train_data = json.load(open(train_ann))
    print(f"  训练集图片数: {len(train_data['images']):,}")
    print(f"  训练集标注数: {len(train_data['annotations']):,}")
    print(f"  验证集图片数: {len(val_data['images']):,}")
    print(f"  验证集标注数: {len(val_data['annotations']):,}")
    print(f"  类别数: {len(val_data['categories'])}")

    print("\n" + "=" * 80)
    print("✓ 所有检查通过！配置正确，可以开始训练。")
    print("=" * 80)

    return True


if __name__ == "__main__":
    try:
        verify_config()
    except Exception as e:
        print(f"\n❌ 验证过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
