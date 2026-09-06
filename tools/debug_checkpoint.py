"""
检查转换后的checkpoint结构
"""
import torch

ckpt_path = '/media/fishyu/fish-14tb-2/YiFei/Grounding_DINO/mmdetection/checkpoints/vit_base_4channel.pth'

print(f"检查checkpoint: {ckpt_path}")
print("="*60)

ckpt = torch.load(ckpt_path, map_location='cpu')

print(f"\n1. Checkpoint顶层keys:")
for key in ckpt.keys():
    print(f"   - {key}")

if 'state_dict' in ckpt:
    state_dict = ckpt['state_dict']
    print(f"\n2. state_dict中的keys数量: {len(state_dict)}")
    print(f"\n3. 前20个keys:")
    for i, key in enumerate(list(state_dict.keys())[:20]):
        value = state_dict[key]
        print(f"   {i+1}. {key:50s} shape: {value.shape}")

    print(f"\n4. 检查关键权重:")
    critical_keys = ['patch_embed', 'pos_embed', 'blocks.0', 'norm']
    for pattern in critical_keys:
        matching = [k for k in state_dict.keys() if pattern in k]
        if matching:
            print(f"   ✓ 找到 '{pattern}': {len(matching)} 个")
            print(f"     示例: {matching[0]}")
        else:
            print(f"   ✗ 未找到 '{pattern}'")
else:
    print("\n❌ checkpoint中没有'state_dict'键")
