import torch
import argparse
from collections import OrderedDict

def parse_args():
    parser = argparse.ArgumentParser(description='Merge DQ-DETR and MMDetection DINO checkpoints')
    parser.add_argument('dino_ckpt', help='Path to the MMDetection DINO (Swin-L) checkpoint')
    parser.add_argument('dqdetr_ckpt', help='Path to the original DQ-DETR checkpoint')
    parser.add_argument('out_file', help='Path to save the merged DQ-DINO checkpoint')
    return parser.parse_args()

def main():
    args = parse_args()

    print(f"Loading DINO checkpoint from: {args.dino_ckpt}")
    dino_ckpt = torch.load(args.dino_ckpt, map_location='cpu')
    
    print(f"Loading DQ-DETR checkpoint from: {args.dqdetr_ckpt}")
    dqdetr_ckpt = torch.load(args.dqdetr_ckpt, map_location='cpu')

    # 1. 准备 MMDetection 格式的 state_dict
    if 'state_dict' in dino_ckpt:
        new_state_dict = dino_ckpt['state_dict']
    else:
        new_state_dict = dino_ckpt  # 有些 ckpt 直接是 state_dict

    # 2. 提取 DQ-DETR 格式的 state_dict
    if 'model' in dqdetr_ckpt:
        dq_source_dict = dqdetr_ckpt['model']
    elif 'state_dict' in dqdetr_ckpt:
        dq_source_dict = dqdetr_ckpt['state_dict']
    else:
        dq_source_dict = dqdetr_ckpt

    print("\nStart merging weights...")
    
    # 3. 定义需要迁移的模块关键词
    # 在 DQ-DETR 源码中，这些模块定义在 Transformer 类中：
    # self.CCM = CategoricalCounting(...)
    # self.CGFE = CGFE(...)
    # self.multiscale = MultiScaleFeature(...)
    target_modules = ['CCM', 'CGFE', 'multiscale']
    
    transferred_keys = []
    
    for k, v in dq_source_dict.items():
        # 原始 DQ-DETR 的 key 通常是 "transformer.CCM.xxx" 或 "module.transformer.CCM.xxx"
        
        # 检查是否包含我们要的目标模块
        is_target_module = any(mod in k for mod in target_modules)
        
        if is_target_module:
            # === Key 映射逻辑 ===
            # MMDetection 中 Transformer 通常也是命名为 "transformer"
            # 因此 key 的后缀基本可以复用，主要是处理前缀差异
            
            # 1. 去除 "module." 前缀 (DDP训练残留)
            new_k = k.replace('module.', '')
            
            # 2. 确保以 "transformer." 开头 (适配 MMDet 结构)
            # 如果原始 key 已经是 transformer.CCM... 则直接使用
            # 如果原始 key 是 backbone... 则忽略
            
            if 'transformer' not in new_k:
                # 理论上 DQ-DETR 的这些模块都在 transformer 下，但也可能有意外
                print(f"Warning: Found target module key but not in transformer: {k}")
                continue

            # 3. 将权重加入到新的 state_dict 中
            # 检查形状是否兼容 (特别是 CGFE 的 channel gate 可能与 embedding dim 有关)
            # 但通常都是 256 dim，应该没问题
            new_state_dict[new_k] = v
            transferred_keys.append(new_k)

    print(f"\nSuccessfully transferred {len(transferred_keys)} keys from DQ-DETR:")
    # 打印前几个看看格式对不对
    for i in range(min(5, len(transferred_keys))):
        print(f" - {transferred_keys[i]}")
    if len(transferred_keys) > 5:
        print(" - ...")

    # 4. 特殊处理：检查 5-scale 兼容性
    # DINO 默认只有 4 个 level_embed，如果 DQ-DINO 是 5 scale，
    # 需要确保 transformer.level_embed 有 5 个 embedding。
    
    level_embed_key = 'transformer.level_embed'
    if level_embed_key in new_state_dict:
        level_embed = new_state_dict[level_embed_key]
        print(f"\nChecking level_embed shape: {level_embed.shape}")
        
        # 如果 DINO 是 4 scale (4, 256)，但你需要 5 scale
        # 这里的策略是：尝试从 DQ-DETR 复制 level_embed (如果是 5 scale)
        # 或者提示用户这一点
        if level_embed.shape[0] < 5:
            print("Warning: DINO checkpoint has fewer than 5 level_embeds.")
            print("Attempting to borrow level_embed from DQ-DETR checkpoint if available...")
            
            dq_level_embed_keys = [k for k in dq_source_dict.keys() if 'level_embed' in k]
            if dq_level_embed_keys:
                dq_level_embed = dq_source_dict[dq_level_embed_keys[0]]
                if dq_level_embed.shape[0] >= 5:
                    new_state_dict[level_embed_key] = dq_level_embed
                    print(f" -> Replaced level_embed with DQ-DETR's version: {dq_level_embed.shape}")
                else:
                    print(" -> DQ-DETR level_embed is also small, skipping replacement.")
            else:
                print(" -> Could not find level_embed in DQ-DETR.")
    else:
        print("\nWarning: 'transformer.level_embed' not found in DINO checkpoint. (Maybe using learned position embedding?)")

    # 5. 保存结果
    # 保持 MMDetection 的 checkpoint 结构 meta + state_dict
    output_dict = dict(state_dict=new_state_dict)
    if 'meta' in dino_ckpt:
        output_dict['meta'] = dino_ckpt['meta']
        
    torch.save(output_dict, args.out_file)
    print(f"\nMerged checkpoint saved to: {args.out_file}")

if __name__ == '__main__':
    main()