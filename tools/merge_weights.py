import torch
from collections import OrderedDict

# ================= 配置路径 =================
# DNTR 的权重 (提供 Backbone 和 Neck)
dntr_ckpt_path = '/media/fishyu/6955024a-ed66-4a86-b94a-687c51c28306/fishyu/YiFei/Grounding_DINO/DNTR/mmdet-dntr/checkpoints/dntr_pretrained_aitod.pth' 
# DETR 的权重 (提供 Transformer Head)
detr_ckpt_path = '/media/fishyu/6955024a-ed66-4a86-b94a-687c51c28306/fishyu/YiFei/Grounding_DINO/mmdetection/checkpoints/detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth'
# 输出路径
output_path = '/media/fishyu/6955024a-ed66-4a86-b94a-687c51c28306/fishyu/YiFei/Grounding_DINO/mmdetection/checkpoints/dn_detr_pretrain.pth'
# ===========================================

def main():
    print(f"Loading DNTR from {dntr_ckpt_path}...")
    dntr = torch.load(dntr_ckpt_path, map_location='cpu')
    dntr_sd = dntr['state_dict'] if 'state_dict' in dntr else dntr

    print(f"Loading DETR from {detr_ckpt_path}...")
    detr = torch.load(detr_ckpt_path, map_location='cpu')
    detr_sd = detr['state_dict'] if 'state_dict' in detr else detr

    new_sd = OrderedDict()
    
    print("Start merging...")

    # 1. 从 DNTR 复制 Backbone 和 Neck
    # 这确保了特征提取器是连贯且预训练过的
    for k, v in dntr_sd.items():
        if k.startswith('backbone.') or k.startswith('neck.'):
            new_sd[k] = v
            print(f"Copied from DNTR: {k}")

    # 2. 从 DETR 复制 Head (Transformer)
    # 注意：需要剔除分类头 (fc_cls)，因为你的类别数(4)与COCO(80)不同
    for k, v in detr_sd.items():
        if k.startswith('bbox_head.'):
            # 剔除分类层权重
            if 'fc_cls' in k:
                print(f"Skipped DETR class head: {k}")
                continue
            
            # 保留回归层 (fc_reg) 和 Transformer 参数
            # 注意: 如果 DETR 是 embed_dim=256，DNTR Neck 输出也必须是 256
            new_sd[k] = v
            # print(f"Copied from DETR: {k}") # 太多了，不打印

    # 3. 保存
    final_dict = dict(meta=dict(), state_dict=new_sd)
    torch.save(final_dict, output_path)
    print(f"\nSuccess! Merged model saved to {output_path}")
    print("Use this file in your config: load_from = '...'")

if __name__ == '__main__':
    main()