#!/bin/bash

# SCANet整合到MMDetection - 快速启动脚本
# 用于水下RGB-Sonar双模态目标检测

set -e

echo "============================================"
echo "SCANet-MMDetection 整合脚本"
echo "============================================"
echo ""

# 路径配置
MMDET_ROOT="/media/fishyu/fish-14tb-2/YiFei/Grounding_DINO/mmdetection"
SCANET_ROOT="/media/fishyu/fish-14tb-2/YiFei/Grounding_DINO/SCANet"
RGB_ROOT="/media/fishyu/fish-14tb-2/YiFei/Dataset/RGBS50_image"
SONAR_ROOT="/media/fishyu/fish-14tb-2/YiFei/Dataset/RGBS50_sonar_corrected"
MERGED_ROOT="/media/fishyu/fish-14tb-2/YiFei/Dataset/RGBS50_merged"

cd $MMDET_ROOT

# 步骤1: 数据预处理
echo "📦 步骤1: 合并RGB和Sonar图像..."
echo "   这可能需要5-10分钟，请耐心等待..."
echo ""

if [ ! -d "$MERGED_ROOT/train" ]; then
    python tools/merge_rgbs_images.py \
        --rgb_root $RGB_ROOT \
        --sonar_root $SONAR_ROOT \
        --output_root $MERGED_ROOT
    echo "✅ 数据合并完成"
else
    echo "⏭️  数据已存在，跳过合并"
fi
echo ""

# 步骤2: 权重转换
echo "🔄 步骤2: 转换SCANet预训练权重..."
SCANET_CKPT="$SCANET_ROOT/checkpoints/SCANet_network_ep0010.pth.tar"
CONVERTED_CKPT="$SCANET_ROOT/checkpoints/scanet_converted.pth"

if [ -f "$SCANET_CKPT" ]; then
    if [ ! -f "$CONVERTED_CKPT" ]; then
        python tools/convert_scanet_ckpt.py \
            --src $SCANET_CKPT \
            --dst $CONVERTED_CKPT
        echo "✅ 权重转换完成"
    else
        echo "⏭️  转换后的权重已存在，跳过转换"
    fi
else
    echo "⚠️  警告: 找不到SCANet checkpoint: $SCANET_CKPT"
    echo "   将从头训练（不使用预训练权重）"
fi
echo ""

# 步骤3: 验证配置
echo "🔍 步骤3: 验证配置文件..."
python tools/misc/print_config.py configs/scanet/scanet_retinanet_rgbs50.py > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ 配置文件验证通过"
else
    echo "❌ 配置文件验证失败，请检查配置"
    exit 1
fi
echo ""

# 步骤4: 训练提示
echo "============================================"
echo "🎉 准备工作完成！现在可以开始训练"
echo "============================================"
echo ""
echo "📝 训练命令:"
echo ""
echo "   单GPU训练:"
echo "   python tools/train.py configs/scanet/scanet_retinanet_rgbs50.py"
echo ""
echo "   多GPU训练 (推荐, 4个GPU):"
echo "   bash tools/dist_train.sh configs/scanet/scanet_retinanet_rgbs50.py 4"
echo ""
echo "   或使用 torchrun:"
echo "   CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \\"
echo "       tools/train.py configs/scanet/scanet_retinanet_rgbs50.py --launcher pytorch"
echo ""
echo "📊 训练监控:"
echo "   tail -f work_dirs/scanet_retinanet_rgbs50/*.log"
echo ""
echo "🧪 评估命令:"
echo "   python tools/test.py \\"
echo "       configs/scanet/scanet_retinanet_rgbs50.py \\"
echo "       work_dirs/scanet_retinanet_rgbs50/epoch_12.pth"
echo ""
echo "📖 更多信息请查看: configs/scanet/README.md"
echo ""
