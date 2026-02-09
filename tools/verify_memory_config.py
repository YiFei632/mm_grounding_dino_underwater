#!/usr/bin/env python3
"""
验证显存优化配置是否能正常运行
在实际训练前进行快速测试
"""

import subprocess
import sys
import time


def check_gpu_memory():
    """检查GPU显存状态"""
    print("=" * 80)
    print("GPU 显存状态检查")
    print("=" * 80)

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True
        )

        lines = result.stdout.strip().split('\n')
        print(f"\n找到 {len(lines)} 个 GPU:\n")

        for line in lines:
            idx, name, total, free, used = line.split(', ')
            total_gb = float(total) / 1024
            free_gb = float(free) / 1024
            used_gb = float(used) / 1024

            print(f"GPU {idx}: {name}")
            print(f"  总显存: {total_gb:.1f} GB")
            print(f"  已使用: {used_gb:.1f} GB ({used_gb/total_gb*100:.1f}%)")
            print(f"  可用: {free_gb:.1f} GB")
            print()

            # 警告：如果GPU已被占用
            if used_gb > 1.0:
                print(f"  ⚠️  警告: GPU {idx} 已有 {used_gb:.1f} GB 被占用")
                print(f"     建议释放GPU或使用其他GPU")
                print()

        return True
    except Exception as e:
        print(f"❌ 无法获取GPU信息: {e}")
        return False


def test_training(config_file, num_gpus=4, num_iters=10):
    """快速测试训练能否启动"""
    print("=" * 80)
    print("训练启动测试")
    print("=" * 80)
    print(f"配置: {config_file}")
    print(f"GPU数量: {num_gpus}")
    print(f"测试迭代: {num_iters}")
    print()
    print("⏳ 启动训练（这可能需要几分钟）...")
    print("-" * 80)

    # 设置环境变量
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'

    # 启动训练
    cmd = [
        "bash", "tools/dist_train.sh",
        config_file,
        str(num_gpus),
        "--cfg-options",
        "max_epochs=1",
        "train_cfg.val_interval=999",
    ]

    print(f"命令: {' '.join(cmd)}\n")

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )

        iter_count = 0
        oom_detected = False

        for line in process.stdout:
            # 打印输出
            print(line, end='')

            # 检测OOM错误
            if 'OutOfMemoryError' in line or 'out of memory' in line.lower():
                oom_detected = True
                print("\n" + "=" * 80)
                print("❌ 检测到显存不足错误 (OOM)")
                print("=" * 80)
                process.terminate()
                break

            # 计算迭代次数
            if 'Epoch [' in line and '] lr:' in line:
                iter_count += 1

                if iter_count >= num_iters:
                    print("\n" + "=" * 80)
                    print(f"✅ 成功完成 {num_iters} 次迭代!")
                    print("=" * 80)
                    process.terminate()
                    break

        process.wait()

        if oom_detected:
            return False
        elif iter_count >= num_iters:
            return True
        else:
            print("\n⚠️  训练提前结束，可能遇到其他错误")
            return False

    except KeyboardInterrupt:
        print("\n\n⚠️  测试被用户中断")
        process.terminate()
        return False
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        return False


def main():
    print("\n" + "=" * 80)
    print("DQDINO 显存优化配置验证")
    print("=" * 80)
    print()

    # 1. 检查GPU
    if not check_gpu_memory():
        print("❌ GPU检查失败，请确保NVIDIA驱动正常")
        return 1

    input("按回车键继续进行训练测试...")
    print()

    # 2. 测试训练
    config = "configs/underwater_grounding_dino/dq_dino/dqdino_swin-l_5scale_o365_memory_optimized.py"
    num_gpus = 4

    if len(sys.argv) > 1:
        config = sys.argv[1]
    if len(sys.argv) > 2:
        num_gpus = int(sys.argv[2])

    success = test_training(config, num_gpus, num_iters=10)

    print("\n" + "=" * 80)
    print("测试结果")
    print("=" * 80)

    if success:
        print("✅ 显存优化配置验证通过!")
        print("\n可以开始完整训练:")
        print(f"  ./train_dqdino_memory_optimized.sh {num_gpus}")
        print()
        return 0
    else:
        print("❌ 显存优化配置验证失败")
        print("\n建议:")
        print("  1. 检查是否有其他进程占用GPU")
        print("  2. 尝试减小batch_size到1")
        print("  3. 进一步减小max_query到1000")
        print("  4. 查看 MEMORY_OPTIMIZATION.md 获取更多帮助")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
