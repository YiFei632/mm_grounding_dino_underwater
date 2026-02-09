#!/usr/bin/env python3
"""
快速测试训练速度并给出优化建议
"""

import subprocess
import time
import re


def test_training_speed(config_file, num_gpus=4, num_iters=50):
    """测试训练速度"""
    print("=" * 80)
    print("训练速度测试")
    print("=" * 80)
    print(f"配置文件: {config_file}")
    print(f"GPU数量: {num_gpus}")
    print(f"测试迭代数: {num_iters}")
    print()
    print("启动训练...")
    print("-" * 80)

    # 构建训练命令
    cmd = [
        "bash", "tools/dist_train.sh",
        config_file,
        str(num_gpus),
        "--cfg-options",
        f"max_epochs=1",
        f"train_cfg.val_interval=999",  # 禁用验证
    ]

    # 启动训练进程
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )

    iter_times = []
    data_times = []

    try:
        for line in process.stdout:
            print(line, end='')

            # 解析时间信息
            # 格式: Epoch [1][50/217786]  lr: 5.0000e-04  time: 1.234  data_time: 0.123
            match = re.search(r'time:\s*([\d.]+).*data_time:\s*([\d.]+)', line)
            if match:
                iter_time = float(match.group(1))
                data_time = float(match.group(2))
                iter_times.append(iter_time)
                data_times.append(data_time)

                if len(iter_times) >= num_iters:
                    print("\n" + "=" * 80)
                    print(f"已完成 {num_iters} 次迭代，停止测试")
                    print("=" * 80)
                    process.terminate()
                    break

    except KeyboardInterrupt:
        print("\n测试被中断")
        process.terminate()

    process.wait()

    if not iter_times:
        print("❌ 无法获取训练速度信息，请检查训练是否正常启动")
        return

    # 分析结果
    avg_iter_time = sum(iter_times) / len(iter_times)
    avg_data_time = sum(data_times) / len(data_times)
    compute_time = avg_iter_time - avg_data_time

    print("\n" + "=" * 80)
    print("速度测试结果")
    print("=" * 80)
    print(f"平均迭代时间: {avg_iter_time:.3f} 秒")
    print(f"平均数据加载时间: {avg_data_time:.3f} 秒 ({avg_data_time/avg_iter_time*100:.1f}%)")
    print(f"平均计算时间: {compute_time:.3f} 秒 ({compute_time/avg_iter_time*100:.1f}%)")

    # 估算训练时间
    train_images = 1742289

    # 从配置中读取参数（简化处理，这里使用估计值）
    print("\n" + "=" * 80)
    print("不同配置的训练时间预估")
    print("=" * 80)

    configs = [
        ("当前配置 (bs=2, 28 epochs)", 2, num_gpus, 28),
        ("优化配置 (bs=4, 12 epochs)", 4, num_gpus, 12),
        ("激进配置 (bs=6, 6 epochs)", 6, num_gpus, 6),
    ]

    for name, bs_per_gpu, gpus, epochs in configs:
        total_bs = bs_per_gpu * gpus
        iters_per_epoch = train_images / total_bs
        total_iters = iters_per_epoch * epochs

        total_seconds = total_iters * avg_iter_time
        days = total_seconds / 86400

        print(f"\n{name}:")
        print(f"  总batch size: {total_bs}")
        print(f"  每epoch迭代: {iters_per_epoch:,.0f}")
        print(f"  总迭代: {total_iters:,.0f}")
        print(f"  预估时间: {days:.1f} 天 ({days*24:.1f} 小时)")

    # 给出优化建议
    print("\n" + "=" * 80)
    print("优化建议")
    print("=" * 80)

    if avg_data_time / avg_iter_time > 0.2:
        print("⚠️  数据加载时间占比过高 ({:.1f}%)".format(avg_data_time/avg_iter_time*100))
        print("   建议:")
        print("   - 增加 num_workers (当前可能是2，建议8-12)")
        print("   - 添加 prefetch_factor=4")
        print("   - 确保数据集在SSD上")
    else:
        print("✓ 数据加载时间正常")

    print()
    if avg_iter_time > 2.0:
        print("⚠️  迭代时间较慢")
        print("   建议:")
        print("   - 启用混合精度训练 (AMP)")
        print("   - 检查是否有不必要的同步操作")
    else:
        print("✓ 迭代时间正常")

    print("\n推荐配置:")
    print("""
train_dataloader = dict(
    batch_size=4,  # 或更大
    num_workers=8,
    prefetch_factor=4,
    pin_memory=True,
)
max_epochs = 12
train_cfg = dict(val_interval=2)

# 可选：启用混合精度
optim_wrapper = dict(
    type='AmpOptimWrapper',
    ...
)
    """)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python test_training_speed.py <config_file> [num_gpus]")
        print()
        print("示例:")
        print("  python test_training_speed.py configs/underwater_grounding_dino/dq_dino/dqdino_pretrained_swin-l_5scale_o365.py 4")
        sys.exit(1)

    config_file = sys.argv[1]
    num_gpus = int(sys.argv[2]) if len(sys.argv) > 2 else 4

    test_training_speed(config_file, num_gpus)
