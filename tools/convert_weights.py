import torch
from collections import OrderedDict

def convert_ucf_weights(src_path, dst_path):
    print(f"Loading source weights from {src_path}...")
    # 原始权重通常保存为 'model' 或直接是 state_dict
    checkpoint = torch.load(src_path, map_location='cpu')
    if 'model' in checkpoint:
        src_state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        src_state_dict = checkpoint['state_dict']
    else:
        src_state_dict = checkpoint

    new_state_dict = OrderedDict()
    
    for k, v in src_state_dict.items():
        old_k = k
        new_k = k

        # -----------------------------------------------
        # 1. Backbone & Enhancer (通常无需修改，或仅需去前缀)
        # -----------------------------------------------
        # 如果原版用了 'module.' 前缀（DDP训练），需要去掉
        if new_k.startswith('module.'):
            new_k = new_k[7:]

        # -----------------------------------------------
        # 2. Encoder -> Neck
        # -----------------------------------------------
        if new_k.startswith('encoder.'):
            # 将 encoder. 替换为 neck.
            new_k = new_k.replace('encoder.', 'neck.')
            
            # 特殊处理 Cross Attention (CFP)
            # 原版: cross_attn_layers.0.xxx -> 新版: cross_attn_raw.0.xxx
            if 'cross_attn_layers.' in new_k:
                new_k = new_k.replace('cross_attn_layers.', 'cross_attn_raw.')
            
            # 原版: cross_attn_layers_fuse.0.xxx -> 新版: cross_attn_enh.0.xxx
            if 'cross_attn_layers_fuse.' in new_k:
                new_k = new_k.replace('cross_attn_layers_fuse.', 'cross_attn_enh.')

        # -----------------------------------------------
        # 3. Decoder -> Head
        # -----------------------------------------------
        # 原版 decoder 模块通常在 'decoder.' 下
        # MMDet 中这些在 'bbox_head.' 下
        elif new_k.startswith('decoder.'):
            # 先替换前缀
            new_k = new_k.replace('decoder.', 'bbox_head.')
            
            # 处理 TransformerDecoder 层级
            # 原版: decoder.decoder.layers -> 新版: bbox_head.transformer.decoder.layers
            # 注意：原版 decoder.decoder 指的是 RTDETRTransformerv2 里的 TransformerDecoder
            if 'bbox_head.decoder.layers' in new_k:
                new_k = new_k.replace('bbox_head.decoder.layers', 'bbox_head.transformer.decoder.layers')
            
            # 处理预测头 (Head)
            # 原版: dec_score_head -> 新版: fc_cls
            if 'dec_score_head' in new_k:
                new_k = new_k.replace('dec_score_head', 'fc_cls')
            
            # 原版: dec_bbox_head -> 新版: fc_reg
            if 'dec_bbox_head' in new_k:
                new_k = new_k.replace('dec_bbox_head', 'fc_reg')

            # 处理 Encoder Selection Head
            # bbox_head.enc_output 保持不变
            # bbox_head.enc_score_head 保持不变
            # bbox_head.enc_bbox_head 保持不变

            # -----------------------------------------------
            # 4. CQI 模块 (Query Interaction)
            # -----------------------------------------------
            # 原版: query_interaction -> 新版: query_interaction
            # 原版: query_interaction2 -> 新版: query_interaction_inv
            if 'query_interaction2' in new_k:
                new_k = new_k.replace('query_interaction2', 'query_interaction_inv')
        
        # 打印部分映射以供检查
        if old_k != new_k:
            # print(f"Mapping: {old_k} -> {new_k}")
            pass

        new_state_dict[new_k] = v

    # 保存为 MMDetection 兼容格式
    # MMDet 期望 checkpoint 包含 'state_dict' 和 'meta' (可选)
    final_dict = dict(state_dict=new_state_dict, meta=dict())
    
    print(f"Saving converted weights to {dst_path}...")
    torch.save(final_dict, dst_path)
    print("Done!")

if __name__ == '__main__':
    # 修改这里的路径
    SRC_PATH = '/media/fishyu/6955024a-ed66-4a86-b94a-687c51c28306/fishyu/YiFei/Grounding_DINO/mmdetection/checkpoints/UCF_DETR_Pretrain_UDD.pth' # 你的原版 .pth 路径
    DST_PATH = '/media/fishyu/6955024a-ed66-4a86-b94a-687c51c28306/fishyu/YiFei/Grounding_DINO/mmdetection/checkpoints/UCF_DETR_Pretrain_UDD_mmdet.pth'
    
    convert_ucf_weights(SRC_PATH, DST_PATH)