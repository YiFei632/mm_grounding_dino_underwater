"""
将RGB和Sonar图像合并为4通道RGBS图像
输入: RGBS50_image (RGB) + RGBS50_sonar_corrected (Sonar)
输出: RGBS50_merged (4-channel)
"""

import os
import cv2
import numpy as np
import json
from tqdm import tqdm
from pathlib import Path
import argparse


def merge_rgbs_dataset(
    rgb_root='/media/fishyu/fish-14tb-2/YiFei/Dataset/RGBS50_image',
    sonar_root='/media/fishyu/fish-14tb-2/YiFei/Dataset/RGBS50_sonar_corrected',
    output_root='/media/fishyu/fish-14tb-2/YiFei/Dataset/RGBS50_merged'
):
    """合并RGB和Sonar数据集"""

    print(f"📁 RGB数据集: {rgb_root}")
    print(f"📁 Sonar数据集: {sonar_root}")
    print(f"📁 输出目录: {output_root}")

    os.makedirs(output_root, exist_ok=True)
    os.makedirs(f'{output_root}/train', exist_ok=True)
    os.makedirs(f'{output_root}/val', exist_ok=True)
    os.makedirs(f'{output_root}/annotations', exist_ok=True)

    for split in ['train', 'val']:
        print(f"\n🔄 处理 {split} 集...")

        # 读取annotation
        rgb_ann_path = f'{rgb_root}/instances_{split}_no_cb.json'
        sonar_ann_path = f'{sonar_root}/instances_{split}_no_cb.json'

        if not os.path.exists(rgb_ann_path):
            print(f"⚠️  警告: 找不到 {rgb_ann_path}，跳过 {split} 集")
            continue

        rgb_ann = json.load(open(rgb_ann_path))

        # 建立Sonar图像映射（通过文件名匹配）
        sonar_map = {}
        if os.path.exists(sonar_ann_path):
            sonar_ann = json.load(open(sonar_ann_path))
            for img_info in sonar_ann['images']:
                # 提取序列名和帧号作为匹配键
                key = Path(img_info['file_name']).stem
                sonar_map[key] = img_info
        else:
            print(f"⚠️  警告: 找不到 {sonar_ann_path}，将使用灰度图作为Sonar通道")

        # 合并图像
        success_count = 0
        fail_count = 0

        for img_info in tqdm(rgb_ann['images'], desc=f'Merging {split}'):
            try:
                # 读取RGB
                rgb_file = img_info['file_name']
                rgb_path = f"{rgb_root}/{split}_images/{rgb_file}"

                if not os.path.exists(rgb_path):
                    print(f"⚠️  找不到RGB文件: {rgb_path}")
                    fail_count += 1
                    continue

                rgb = cv2.imread(rgb_path)
                if rgb is None:
                    print(f"⚠️  无法读取RGB文件: {rgb_path}")
                    fail_count += 1
                    continue

                # 读取对应的Sonar（通过文件名匹配）
                key = Path(rgb_file).stem

                if key in sonar_map and sonar_map[key]:
                    sonar_file = sonar_map[key]['file_name']
                    sonar_path = f"{sonar_root}/{split}_images/{sonar_file}"

                    if os.path.exists(sonar_path):
                        sonar = cv2.imread(sonar_path, cv2.IMREAD_GRAYSCALE)
                        if sonar is None:
                            print(f"⚠️  无法读取Sonar文件: {sonar_path}，使用灰度RGB作为替代")
                            sonar = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
                    else:
                        print(f"⚠️  找不到Sonar文件: {sonar_path}，使用灰度RGB作为替代")
                        sonar = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
                else:
                    # 没有对应的Sonar，使用RGB的灰度图作为替代
                    sonar = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

                # 调整Sonar尺寸以匹配RGB
                if sonar.shape[:2] != rgb.shape[:2]:
                    sonar = cv2.resize(sonar, (rgb.shape[1], rgb.shape[0]))

                # 合并为4通道 [R, G, B, S]
                rgbs = np.dstack([rgb, sonar])

                # 保存为npy格式（支持4通道）
                output_file = f"{output_root}/{split}/{Path(rgb_file).stem}.npy"
                np.save(output_file, rgbs.astype(np.uint8))

                success_count += 1

            except Exception as e:
                print(f"❌ 处理失败 {rgb_file}: {e}")
                fail_count += 1
                continue

        print(f"✅ {split} 集处理完成: 成功 {success_count} 个，失败 {fail_count} 个")

        # 修改annotation中的file_name为.npy
        for img_info in rgb_ann['images']:
            original_name = img_info['file_name']
            img_info['file_name'] = Path(original_name).stem + '.npy'

        # 保存修改后的annotation
        output_ann_path = f'{output_root}/annotations/instances_{split}.json'
        with open(output_ann_path, 'w') as f:
            json.dump(rgb_ann, f, indent=2)

        print(f"💾 保存annotation: {output_ann_path}")

    print(f"\n🎉 所有数据合并完成!")
    print(f"📂 输出目录: {output_root}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='合并RGB和Sonar数据集')
    parser.add_argument('--rgb_root', type=str,
                        default='/media/fishyu/fish-14tb-2/YiFei/Dataset/RGBS50_image',
                        help='RGB数据集根目录')
    parser.add_argument('--sonar_root', type=str,
                        default='/media/fishyu/fish-14tb-2/YiFei/Dataset/RGBS50_sonar_corrected',
                        help='Sonar数据集根目录')
    parser.add_argument('--output_root', type=str,
                        default='/media/fishyu/fish-14tb-2/YiFei/Dataset/RGBS50_merged',
                        help='输出目录')

    args = parser.parse_args()

    merge_rgbs_dataset(
        rgb_root=args.rgb_root,
        sonar_root=args.sonar_root,
        output_root=args.output_root
    )
