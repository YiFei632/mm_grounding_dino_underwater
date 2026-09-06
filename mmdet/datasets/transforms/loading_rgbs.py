"""
自定义数据预处理器，支持加载.npy格式的4通道图像
"""

import torch
import numpy as np
from mmdet.registry import TRANSFORMS
from mmcv.transforms import BaseTransform


@TRANSFORMS.register_module()
class LoadRGBSImageFromFile(BaseTransform):
    """
    从文件加载RGB-Sonar 4通道图像
    支持.npy格式（4通道）和常规图像格式（3通道，自动补0）
    """

    def __init__(self, to_float32=False, color_type='color',
                 mean=[123.675, 116.28, 103.53, 127.5],
                 std=[58.395, 57.12, 57.375, 30.0],
                 **kwargs):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def transform(self, results):
        """
        Args:
            results (dict): 包含'img_path'或'img'字段的结果字典

        Returns:
            dict: 添加了'img', 'img_shape', 'ori_shape'字段
        """
        filename = results.get('img_path', None)

        if filename is None:
            # 如果已经加载了图像，直接返回
            if 'img' in results:
                return results
            raise ValueError("Neither 'img_path' nor 'img' found in results")

        # 检查文件格式
        if filename.endswith('.npy'):
            # 加载4通道npy文件
            img = np.load(filename)  # shape: (H, W, 4)

            if img.ndim == 3 and img.shape[2] == 4:
                # 正确的4通道格式
                pass
            elif img.ndim == 3 and img.shape[2] == 3:
                # 3通道，补一个0通道
                zero_channel = np.zeros((img.shape[0], img.shape[1], 1), dtype=img.dtype)
                img = np.concatenate([img, zero_channel], axis=2)
            else:
                raise ValueError(f"Unexpected npy shape: {img.shape}")

        else:
            # 使用mmcv加载常规图像（jpg/png等）
            import mmcv
            img = mmcv.imread(filename, flag=self.color_type)

            # 补充第4通道（全0或使用灰度）
            if img.ndim == 3 and img.shape[2] == 3:
                # 使用RGB的灰度作为第4通道
                gray = np.mean(img, axis=2, keepdims=True).astype(img.dtype)
                img = np.concatenate([img, gray], axis=2)
            elif img.ndim == 2:
                # 灰度图，扩展为4通道
                img = np.stack([img, img, img, img], axis=2)

        # 转换为float32并归一化
        if self.to_float32:
            img = img.astype(np.float32)
            # 归一化: (img - mean) / std
            img = (img - self.mean.reshape(1, 1, 4)) / self.std.reshape(1, 1, 4)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]

        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(to_float32={self.to_float32})'
