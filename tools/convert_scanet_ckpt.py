"""
将SCANet checkpoint转换为MMDetection格式
"""

import torch
import argparse
from collections import OrderedDict
import sys
import os


# 添加SCANet路径以解决导入问题
SCANET_ROOT = '/media/fishyu/fish-14tb-2/YiFei/Grounding_DINO/SCANet'
if os.path.exists(SCANET_ROOT):
    sys.path.insert(0, SCANET_ROOT)


def convert_scanet_checkpoint(src_path, dst_path, verbose=True):
    """
    将SCANet的跟踪模型权重转换为MMDetection检测模型格式

    Args:
        src_path: SCANet原始checkpoint路径
        dst_path: 输出的MMDetection格式checkpoint路径
        verbose: 是否打印详细信息
    """
    print(f"📂 加载SCANet checkpoint: {src_path}")

    # 加载SCANet checkpoint，使用weights_only=False允许加载完整对象
    try:
        ckpt = torch.load(src_path, map_location='cpu', weights_only=False)
    except TypeError:
        # 旧版本torch不支持weights_only参数
        ckpt = torch.load(src_path, map_location='cpu')

    # SCANet的权重可能在不同的键下
    if 'net' in ckpt:
        scanet_state = ckpt['net']
        print("   检测到'net'键")
    elif 'state_dict' in ckpt:
        scanet_state = ckpt['state_dict']
        print("   检测到'state_dict'键")
    elif 'model' in ckpt:
        scanet_state = ckpt['model']
        print("   检测到'model'键")
    else:
        scanet_state = ckpt
        print("   直接使用checkpoint")

    print(f"   原始权重数量: {len(scanet_state)}")

    # 权重映射
    mmdet_state = OrderedDict()
    skipped_keys = []
    converted_keys = []

    for key, value in scanet_state.items():
        new_key = key
        skip = False

        # 映射规则
        # 1. 移除'backbone.'前缀（如果存在）
        if new_key.startswith('backbone.'):
            new_key = new_key.replace('backbone.', '')

        # 2. 跳过检测头相关权重（不兼容MMDetection的检测头）
        if any(x in key for x in ['box_head', 'sonar_head', 'head']):
            skip = True
            skipped_keys.append(key)

        # 3. 处理patch_embed相关
        # SCANet可能是3通道，我们需要4通道，这部分权重可能需要手动处理
        if 'patch_embed.proj.weight' in key:
            # 检查输入通道数
            if value.shape[1] == 3:
                print(f"   ⚠️  检测到3通道patch_embed，将扩展为4通道")
                # 复制最后一个通道作为第4通道的初始化
                extra_channel = value[:, -1:, :, :].clone()
                value = torch.cat([value, extra_channel], dim=1)
                print(f"      新shape: {value.shape}")

        # 4. 添加到新的state_dict
        if not skip:
            mmdet_state[new_key] = value
            converted_keys.append((key, new_key))

    # 保存为MMDetection格式
    output_dict = {
        'state_dict': mmdet_state,
        'meta': {
            'converted_from': src_path,
            'original_keys': len(scanet_state),
            'converted_keys': len(mmdet_state),
            'skipped_keys': len(skipped_keys)
        }
    }

    torch.save(output_dict, dst_path)

    # 打印统计信息
    print(f"\n✅ 转换完成!")
    print(f"   源文件: {src_path}")
    print(f"   目标文件: {dst_path}")
    print(f"   原始权重数: {len(scanet_state)}")
    print(f"   转换权重数: {len(mmdet_state)}")
    print(f"   跳过权重数: {len(skipped_keys)}")

    if verbose and skipped_keys:
        print(f"\n📋 跳过的权重 (前10个):")
        for key in skipped_keys[:10]:
            print(f"   - {key}")
        if len(skipped_keys) > 10:
            print(f"   ... 还有 {len(skipped_keys) - 10} 个")

    if verbose and converted_keys:
        print(f"\n📋 转换的权重示例 (前5个):")
        for old_key, new_key in converted_keys[:5]:
            if old_key != new_key:
                print(f"   {old_key} -> {new_key}")
            else:
                print(f"   {old_key}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='转换SCANet权重到MMDetection格式')
    parser.add_argument('--src', type=str, required=True,
                        help='SCANet checkpoint路径')
    parser.add_argument('--dst', type=str, required=True,
                        help='输出的MMDetection checkpoint路径')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='打印详细信息')

    args = parser.parse_args()

    convert_scanet_checkpoint(args.src, args.dst, args.verbose)

    print("\n💡 使用提示:")
    print("   在config中通过以下方式加载:")
    print(f"   load_from = '{args.dst}'")
