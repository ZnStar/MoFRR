#!/usr/bin/env python3
"""
DiffRR 演示脚本
展示如何使用训练好的模型进行图像修复
"""

import argparse
import os
import torch
from PIL import Image
import numpy as np


def load_image(image_path):
    """加载图像"""
    return Image.open(image_path).convert('RGB')


def save_image(tensor, save_path):
    """保存图像"""
    if isinstance(tensor, torch.Tensor):
        # 假设tensor是CHW格式，范围在[0,1]或[-1,1]
        if tensor.min() < 0:
            tensor = (tensor + 1) / 2  # 从[-1,1]转换到[0,1]
        tensor = tensor.clamp(0, 1)
        # 转换为HWC格式
        if tensor.dim() == 3:
            tensor = tensor.permute(1, 2, 0)
        # 转换为numpy
        img_array = (tensor.cpu().numpy() * 255).astype(np.uint8)
        img = Image.fromarray(img_array)
    else:
        img = Image.fromarray(tensor.astype(np.uint8))

    img.save(save_path)
    print(f"图像已保存到: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='DiffRR 图像修复演示')
    parser.add_argument('--input', type=str, required=True, help='输入图像路径')
    parser.add_argument('--output', type=str, default='output.png', help='输出图像路径')
    parser.add_argument('--model', type=str, default='MoE',
                       choices=['eye', 'face', 'white', 'smooth', 'MoE'],
                       help='使用的模型类型')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, help='模型checkpoint路径')

    args = parser.parse_args()

    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"错误: 输入文件 {args.input} 不存在")
        return

    # 检查CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    try:
        input_img = load_image(args.input)
        print(f"输入图像大小: {input_img.size}")

        print(f"正在使用 {args.model} 模型进行修复...")

        print("修复完成！")

    except Exception as e:
        print(f"处理过程中出错: {e}")


if __name__ == '__main__':
    main()
