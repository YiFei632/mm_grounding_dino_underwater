"""
将ImageNet预训练的ViT-Base (3通道) 扩展为4通道 (RGB+Sonar)
用于SCANet检测任务
"""

import torch
import argparse


def extend_vit_to_4channel(src_path, dst_path, method='copy_last'):
    """
    将3通道ViT扩展到4通道

    Args:
        src_path: 原始3通道ViT checkpoint路径
        dst_path: 输出4通道checkpoint路径
        method: 扩展方法
            - 'copy_last': 复制最后一个通道
            - 'average': 使用RGB三通道的平均
            - 'scaled': 缩放后复制，保持总输入能量
    """
    print(f"📂 加载3通道ViT checkpoint: {src_path}")

    # 加载checkpoint
    ckpt = torch.load(src_path, map_location='cpu')

    # 获取state_dict（不同来源的checkpoint结构可能不同）
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
        print("   使用 'state_dict' 键")
    elif 'model' in ckpt:
        state_dict = ckpt['model']
        print("   使用 'model' 键")
    else:
        state_dict = ckpt
        print("   直接使用checkpoint")

    print(f"   原始权重数量: {len(state_dict)}")

    # 查找patch embedding权重的key名称
    patch_embed_key = None
    for key in state_dict.keys():
        if 'patch_embed' in key and 'proj.weight' in key:
            patch_embed_key = key
            break

    if patch_embed_key is None:
        # 尝试其他可能的名称
        for key in state_dict.keys():
            if 'conv1' in key or 'stem' in key:
                patch_embed_key = key
                break

    if patch_embed_key is None:
        print("❌ 错误: 找不到patch embedding权重")
        print("   可用的keys:")
        for key in list(state_dict.keys())[:20]:
            print(f"     - {key}")
        return

    print(f"✅ 找到patch embedding: {patch_embed_key}")

    # 获取patch embedding权重
    patch_weight = state_dict[patch_embed_key]
    print(f"   原始shape: {patch_weight.shape}")

    # 检查是否是3通道输入
    if patch_weight.shape[1] != 3:
        print(f"⚠️  警告: 输入通道不是3，而是{patch_weight.shape[1]}")
        print("   继续处理...")

    # 扩展到4通道
    if method == 'copy_last':
        # 方法1: 复制最后一个通道（B通道）
        print("📝 使用方法: 复制最后一个通道")
        new_channel = patch_weight[:, -1:, :, :]  # (out_channels, 1, kernel_h, kernel_w)
        patch_weight_4ch = torch.cat([patch_weight, new_channel], dim=1)

    elif method == 'average':
        # 方法2: 使用RGB三通道的平均值
        print("📝 使用方法: RGB三通道平均")
        new_channel = patch_weight.mean(dim=1, keepdim=True)
        patch_weight_4ch = torch.cat([patch_weight, new_channel], dim=1)

    elif method == 'scaled':
        # 方法3: 缩放复制，保持总输入能量不变
        print("📝 使用方法: 缩放复制（能量守恒）")
        # RGB通道缩放到0.75，新通道占0.25
        rgb_scaled = patch_weight * 0.75
        new_channel = patch_weight[:, -1:, :, :] * 0.25
        patch_weight_4ch = torch.cat([rgb_scaled, new_channel], dim=1)

    else:
        raise ValueError(f"Unknown method: {method}")

    print(f"   扩展后shape: {patch_weight_4ch.shape}")

    # 替换权重
    state_dict[patch_embed_key] = patch_weight_4ch

    # 处理position embedding（如果存在且需要调整）
    # 注意: position embedding通常不需要修改，因为它是与spatial位置相关，与通道数无关

    # 保存新的checkpoint
    output = {
        'state_dict': state_dict,
        'meta': {
            'source': src_path,
            'method': method,
            'channels': 4,
            'modified_keys': [patch_embed_key]
        }
    }

    torch.save(output, dst_path)

    print(f"\n✅ 转换完成!")
    print(f"   源文件: {src_path}")
    print(f"   目标文件: {dst_path}")
    print(f"   扩展方法: {method}")
    print(f"   修改的key: {patch_embed_key}")
    print(f"   新权重数量: {len(state_dict)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='扩展ViT到4通道')
    parser.add_argument('--src', type=str, required=True,
                        help='源ViT checkpoint路径 (3通道)')
    parser.add_argument('--dst', type=str, required=True,
                        help='输出checkpoint路径 (4通道)')
    parser.add_argument('--method', type=str, default='copy_last',
                        choices=['copy_last', 'average', 'scaled'],
                        help='扩展方法')

    args = parser.parse_args()

    extend_vit_to_4channel(args.src, args.dst, args.method)

    print("\n💡 使用提示:")
    print("   在配置文件中这样加载:")
    print(f"   init_cfg=dict(")
    print(f"       type='Pretrained',")
    print(f"       checkpoint='{args.dst}',")
    print(f"       prefix=''")
    print(f"   )")
