# SCANet for Underwater Detection

## 模型介绍

SCANet (Sonar-Camera Attention Network) 是为水下RGB-Sonar双模态目标检测设计的模型。本实现将SCANet从原始的跟踪任务改造为检测任务，并整合进MMDetection框架。

### 主要特点

- **双模态融合**: 通过SCAM模块融合RGB和Sonar特征
- **ViT Backbone**: 基于Vision Transformer的特征提取
- **简化输入**: 采用4通道输入 (RGB + Sonar)
- **标准检测头**: 使用RetinaNet检测头

## 环境配置

基于MMDetection环境，额外需要：

```bash
pip install numpy opencv-python pycocotools
```

## 数据准备

### 步骤1: 合并RGB和Sonar图像

```bash
cd /media/fishyu/fish-14tb-2/YiFei/Grounding_DINO/mmdetection

python tools/merge_rgbs_images.py \
    --rgb_root /media/fishyu/fish-14tb-2/YiFei/Dataset/RGBS50_image \
    --sonar_root /media/fishyu/fish-14tb-2/YiFei/Dataset/RGBS50_sonar_corrected \
    --output_root /media/fishyu/fish-14tb-2/YiFei/Dataset/RGBS50_merged
```

生成的数据格式：
```
RGBS50_merged/
├── train/
│   ├── seq1_frame001.npy  (4-channel: RGB + Sonar)
│   └── ...
├── val/
│   └── ...
└── annotations/
    ├── instances_train.json
    └── instances_val.json
```

### 步骤2: 转换SCANet预训练权重

```bash
python tools/convert_scanet_ckpt.py \
    --src /media/fishyu/fish-14tb-2/YiFei/Grounding_DINO/SCANet/checkpoints/SCANet_network_ep0010.pth.tar \
    --dst /media/fishyu/fish-14tb-2/YiFei/Grounding_DINO/SCANet/checkpoints/scanet_converted.pth
```

## 训练

### 单GPU训练

```bash
python tools/train.py configs/scanet/scanet_retinanet_rgbs50.py
```

### 多GPU训练 (推荐)

```bash
# 4个GPU
bash tools/dist_train.sh configs/scanet/scanet_retinanet_rgbs50.py 4

# 或使用torchrun
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    tools/train.py configs/scanet/scanet_retinanet_rgbs50.py --launcher pytorch
```

### 训练监控

```bash
# 查看日志
tail -f work_dirs/scanet_retinanet_rgbs50/*.log

# TensorBoard
tensorboard --logdir work_dirs/scanet_retinanet_rgbs50
```

## 测试和评估

### 在验证集上评估

```bash
python tools/test.py \
    configs/scanet/scanet_retinanet_rgbs50.py \
    work_dirs/scanet_retinanet_rgbs50/epoch_12.pth \
    --work-dir work_dirs/scanet_retinanet_rgbs50/eval
```

### 查看评估结果

```bash
# COCO指标
cat work_dirs/scanet_retinanet_rgbs50/eval/*.log.json
```

### 可视化预测结果

```bash
python tools/test.py \
    configs/scanet/scanet_retinanet_rgbs50.py \
    work_dirs/scanet_retinanet_rgbs50/epoch_12.pth \
    --show-dir work_dirs/scanet_retinanet_rgbs50/vis_results
```

## 配置说明

### 模型配置

- **Backbone**: SCANetViT
  - 输入: 4通道 (RGB + Sonar)
  - Patch size: 16
  - Embed dim: 768
  - Depth: 12层
  - SCAM融合层: 第3、6、9层

- **Neck**: FPN
  - 输入通道: 768 (来自ViT)
  - 输出通道: 256
  - 输出尺度: 5个

- **Head**: RetinaNetHead
  - 类别数: 1 (underwater object)
  - Anchor scales: [4]
  - Anchor ratios: [0.5, 1.0, 2.0]

### 训练配置

- **Epochs**: 12
- **Batch size**: 8 (单GPU)
- **Optimizer**: AdamW (lr=0.0001, weight_decay=0.0001)
- **Learning rate schedule**:
  - Warmup: 500 iterations
  - Decay: 第8和11个epoch
- **Input size**: 640x640

## 性能预期

在RGBS50数据集上:

- **训练时间**: 约2-4小时 (4 GPUs)
- **推理速度**: ~15-20 FPS
- **预期mAP**: 0.45-0.65 (取决于数据质量和训练设置)

## 文件结构

```
configs/scanet/
├── scanet_retinanet_rgbs50.py      # 主配置
├── _base_/
│   ├── models/
│   │   └── scanet_retinanet.py     # 模型配置
│   ├── datasets/
│   │   └── rgbs50.py               # 数据集配置
│   └── schedules/
│       └── schedule_1x.py          # 训练配置
└── README.md                        # 本文件

mmdet/models/backbones/
├── scanet_vit.py                    # SCANet ViT Backbone
└── scanet_components/
    ├── patch_embed.py
    ├── blocks.py
    └── scam.py

mmdet/datasets/transforms/
└── loading_rgbs.py                  # 4通道图像加载

tools/
├── merge_rgbs_images.py             # 数据预处理
└── convert_scanet_ckpt.py           # 权重转换
```

## 常见问题

### Q1: 数据加载失败

**错误**: `ValueError: Unexpected npy shape`

**解决**: 检查.npy文件格式，确保是 (H, W, 4) 的shape

```python
import numpy as np
img = np.load('xxx.npy')
print(img.shape)  # 应该是 (H, W, 4)
```

### Q2: GPU内存不足

**错误**: `CUDA out of memory`

**解决**: 减小batch size

```python
# 在配置文件中修改
train_dataloader = dict(
    batch_size=4,  # 从8减小到4
    ...
)
```

### Q3: 预训练权重加载失败

**错误**: `Missing keys / Unexpected keys`

**解决**: 这是正常的，因为检测头与跟踪头不兼容。只要backbone权重加载成功即可。

### Q4: 如何使用不同的检测头

可以替换为其他检测头，例如FCOS:

```python
# 在配置文件中修改
model = dict(
    type='FCOS',  # 替换为FCOS
    bbox_head=dict(
        type='FCOSHead',
        num_classes=1,
        ...
    )
)
```

## 引用

如果使用本代码，请引用原始SCANet论文:

```bibtex
@article{scanet2024,
  title={SCANet: RGB-Sonar Attention Network for Underwater Tracking},
  journal={arXiv preprint},
  year={2024}
}
```

## 联系方式

如有问题，请提交issue或联系开发者。
