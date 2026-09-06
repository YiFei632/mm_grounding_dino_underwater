# SCANet方案1使用指南：使用ImageNet预训练ViT

## 步骤1: 下载ViT-Base预训练权重

### 选项A: timm预训练（推荐）
```bash
cd /media/fishyu/fish-14tb-2/YiFei/Grounding_DINO/mmdetection/checkpoints

# 下载ViT-Base patch16 224
wget https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth

# 或者使用镜像
wget https://huggingface.co/timm/vit_base_patch16_224.augreg_in21k/resolve/main/pytorch_model.bin
```

### 选项B: torchvision预训练
```python
import torchvision.models as models
import torch

# 下载并保存
model = models.vit_b_16(pretrained=True)
torch.save(model.state_dict(), 'checkpoints/vit_b_16_torchvision.pth')
```

### 选项C: 如果已有其他ViT checkpoint
如果你已经有其他来源的ViT checkpoint，直接使用即可。

---

## 步骤2: 扩展到4通道

运行转换脚本：

```bash
cd /media/fishyu/fish-14tb-2/YiFei/Grounding_DINO/mmdetection

# 方法1: 复制最后一个通道（推荐）
python tools/extend_vit_to_4channel.py \
    --src checkpoints/jx_vit_base_p16_224-80ecf9dd.pth \
    --dst checkpoints/vit_base_4channel.pth \
    --method copy_last

# 方法2: 使用RGB平均（更保守）
python tools/extend_vit_to_4channel.py \
    --src checkpoints/jx_vit_base_p16_224-80ecf9dd.pth \
    --dst checkpoints/vit_base_4channel_avg.pth \
    --method average

# 方法3: 缩放复制（能量守恒）
python tools/extend_vit_to_4channel.py \
    --src checkpoints/jx_vit_base_p16_224-80ecf9dd.pth \
    --dst checkpoints/vit_base_4channel_scaled.pth \
    --method scaled
```

**三种方法对比**:
- `copy_last`: 直接复制B通道，简单直接
- `average`: 使用RGB平均值，更平滑
- `scaled`: 保持总输入能量，理论上更合理

建议先用`copy_last`，如果效果不好再尝试其他方法。

---

## 步骤3: 修改配置文件中的checkpoint路径

打开 `configs/scanet/scanet_retinanet_rgbs50_with_pretrain.py`

找到第47行附近：
```python
init_cfg=dict(
    type='Pretrained',
    checkpoint='checkpoints/vit_base_4channel.pth',  # ← 修改这里
    prefix=''
)
```

改为你实际的checkpoint路径：
```python
checkpoint='/media/fishyu/fish-14tb-2/YiFei/Grounding_DINO/mmdetection/checkpoints/vit_base_4channel.pth'
```

---

## 步骤4: 开始训练

```bash
cd /media/fishyu/fish-14tb-2/YiFei/Grounding_DINO/mmdetection

# 单GPU
python tools/train.py configs/scanet/scanet_retinanet_rgbs50_with_pretrain.py

# 多GPU (推荐)
bash tools/dist_train.sh configs/scanet/scanet_retinanet_rgbs50_with_pretrain.py 4
```

---

## 关键配置变化

与从头训练相比，使用预训练的配置有以下变化：

| 配置项 | 从头训练 | 使用预训练 |
|--------|----------|-----------|
| 学习率 | 0.0001 | 0.00005 (降低50%) |
| Batch size | 8 | 16 (可以增大) |
| Epochs | 20 | 30 |
| LR衰减点 | [8, 11] | [20, 27] |

---

## 验证权重加载是否成功

训练开始时，查看日志中是否有：
```
load checkpoint from: checkpoints/vit_base_4channel.pth
missing keys: []  # 或者很少的keys
unexpected keys: []
```

如果有很多`missing keys`或`unexpected keys`，说明key名称不匹配，需要调整。

---

## 预期效果提升

使用ImageNet预训练后，预期改进：
- **收敛速度**: 前5个epoch就能看到明显效果（从头训练需要10+ epoch）
- **最终mAP**: 提升10-20个点（例如从0.35提升到0.45-0.55）
- **训练稳定性**: 更稳定，loss曲线更平滑

---

## 故障排查

### 问题1: 转换脚本找不到patch_embed key
```
❌ 错误: 找不到patch embedding权重
```

**解决**: 手动查看checkpoint结构
```python
import torch
ckpt = torch.load('xxx.pth', map_location='cpu')
if 'state_dict' in ckpt:
    keys = ckpt['state_dict'].keys()
else:
    keys = ckpt.keys()

for k in keys:
    print(k)
```

找到包含`conv`或`proj`的key，修改转换脚本中的`patch_embed_key`查找逻辑。

### 问题2: 加载权重时shape不匹配
```
Error: size mismatch for patch_embed.proj.weight
```

**解决**: 检查转换后的shape是否正确
```python
ckpt = torch.load('vit_base_4channel.pth', map_location='cpu')
print(ckpt['state_dict']['patch_embed.proj.weight'].shape)
# 应该是 (768, 4, 16, 16)，不是 (768, 3, 16, 16)
```

### 问题3: 训练效果仍然不好

可能的原因和解决方案：
1. **学习率太高**: 降低到0.00001
2. **数据增强不够**: 添加ColorJitter、RandomErasing等
3. **第4通道初始化不好**: 尝试其他扩展方法（average/scaled）
4. **输入尺寸太小**: 尝试从224增加到384（需要插值position embedding）

---

## 高级选项：不同分辨率

如果想使用更大的输入尺寸（如384x384）：

1. 修改配置中的`img_size`:
```python
backbone=dict(
    img_size=384,  # 从224改为384
    ...
)
```

2. 插值position embedding（在转换脚本中添加）：
```python
# 假设pos_embed的shape是 (1, 196, 768) for 224x224
# 需要插值到 (1, 576, 768) for 384x384

import torch.nn.functional as F

pos_embed = state_dict['pos_embed']  # (1, N, D)
old_size = int((pos_embed.shape[1]) ** 0.5)  # 14 for 224
new_size = img_size // patch_size  # 24 for 384

# Reshape and interpolate
pos_embed_reshaped = pos_embed.reshape(1, old_size, old_size, -1).permute(0, 3, 1, 2)
pos_embed_new = F.interpolate(pos_embed_reshaped, size=(new_size, new_size), mode='bicubic')
pos_embed_new = pos_embed_new.permute(0, 2, 3, 1).flatten(1, 2)

state_dict['pos_embed'] = pos_embed_new
```

---

## 总结

方案1的优势：
- ✅ 充分利用ImageNet大规模预训练
- ✅ 实现简单，只需要一个转换脚本
- ✅ 效果提升显著
- ✅ 收敛速度快

现在你可以：
1. 下载ViT权重
2. 运行转换脚本
3. 修改配置文件中的checkpoint路径
4. 开始训练

Good luck! 🚀
