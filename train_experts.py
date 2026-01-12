#!/usr/bin/env python3

import os
import sys
import argparse
import subprocess
from pathlib import Path


def check_environment():
    """检查训练环境"""
    print("=== 环境检查 ===")

    # 检查PyTorch
    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("警告: 未检测到CUDA")
    except ImportError:
        print("警告: 无法导入PyTorch，请确保已正确安装")
        return False

    # 检查必要的文件
    required_files = [
        'configs/RetouchingOne_eye_uncond.yml',
        'configs/RetouchingOne_face_uncond.yml',
        'configs/RetouchingOne_white_uncond.yml',
        'configs/RetouchingOne_smooth_uncond.yml',
        'MoE/RetouchingOne_MoE_uncond.yml'
    ]

    print("\n=== 文件检查 ===")
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} (缺失)")
            missing_files.append(file_path)

    if missing_files:
        print(f"\n错误: 缺少必要的配置文件: {missing_files}")
        return False

    return True


def run_command(cmd, description):
    """运行命令并显示结果"""
    print(f"\n=== {description} ===")
    print(f"执行命令: {cmd}")

    try:
        result = subprocess.run(cmd, shell=True, check=True,
                              capture_output=True, text=True)
        print("命令执行成功!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"命令执行失败 (退出码: {e.returncode})")
        if e.stdout:
            print(f"标准输出:\n{e.stdout}")
        if e.stderr:
            print(f"标准错误:\n{e.stderr}")
        return False


def train_expert_model(config_file, expert_name, gpu_id=None):
    """训练单个专家模型"""
    cmd = "python train.py --config " + config_file
    if gpu_id is not None:
        cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} " + cmd

    description = f"训练{expert_name}专家模型"
    return run_command(cmd, description)


def check_expert_models():
    """检查专家模型是否已训练完成"""
    print("\n=== 检查专家模型状态 ===")

    expert_configs = {
        'eye': 'ckpt/eye/model_latest_eye.pth.tar',
        'face': 'ckpt/face/model_latest_face.pth.tar',
        'white': 'ckpt/white/model_latest_white.pth.tar',
        'smooth': 'ckpt/smooth/model_latest_smooth.pth.tar'
    }

    all_ready = True
    for expert, model_path in expert_configs.items():
        if os.path.exists(model_path):
            print(f"✓ {expert}专家模型已存在: {model_path}")
        else:
            print(f"✗ {expert}专家模型不存在: {model_path}")
            all_ready = False

    return all_ready


def train_moe_model(gpu_id=None):
    """训练MoE融合模型"""
    cmd = "python MoE_train.py --config MoE/RetouchingOne_MoE_uncond.yml"
    if gpu_id is not None:
        cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} " + cmd

    description = "训练MoE融合模型"
    return run_command(cmd, description)


def evaluate_models():
    """评估训练好的模型"""
    print("\n=== 模型评估 ===")

    # 评估专家模型
    expert_configs = [
        ('configs/RetouchingOne_eye_uncond.yml', '眼睛专家'),
        ('configs/RetouchingOne_face_uncond.yml', '脸部专家'),
        ('configs/RetouchingOne_white_uncond.yml', '美白专家'),
        ('configs/RetouchingOne_smooth_uncond.yml', '磨皮专家')
    ]

    print("评估专家模型...")
    for config_file, name in expert_configs:
        cmd = f"python evaluate.py --config {config_file}"
        run_command(cmd, f"评估{name}")

    # 评估MoE模型
    print("\n评估MoE融合模型...")
    cmd = "python MoE_test.py --config MoE/RetouchingOne_MoE_uncond.yml"
    run_command(cmd, "评估MoE融合模型")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='MoFRR 专家模型训练脚本')
    parser.add_argument('--stage', type=str, choices=['1', '2', 'all'],
                       default='all', help='训练阶段: 1(专家模型), 2(MoE融合), all(全部)')
    parser.add_argument('--gpu', type=str, help='指定GPU设备ID，例如: 0,1 或 0')
    parser.add_argument('--skip_check', action='store_true',
                       help='跳过环境和文件检查')
    parser.add_argument('--evaluate', action='store_true',
                       help='训练完成后进行评估')

    args = parser.parse_args()

    print("MoFRR 专家模型训练脚本")
    print("=" * 50)

    # 环境检查
    if not args.skip_check:
        if not check_environment():
            print("\n环境检查失败，请解决上述问题后重试")
            sys.exit(1)

    # 解析GPU设置
    gpu_ids = None
    if args.gpu:
        gpu_ids = [int(x.strip()) for x in args.gpu.split(',')]

    # 第一阶段
    if args.stage in ['1', 'all']:
        print("\n" + "=" * 50)
        print("第一阶段:训练")
        print("=" * 50)

        expert_configs = [
            ('configs/RetouchingOne_eye_uncond.yml', '眼睛'),
            ('configs/RetouchingOne_face_uncond.yml', '脸部'),
            ('configs/RetouchingOne_white_uncond.yml', '美白'),
            ('configs/RetouchingOne_smooth_uncond.yml', '磨皮')
        ]

        success_count = 0
        for i, (config_file, expert_name) in enumerate(expert_configs):
            gpu_id = gpu_ids[i % len(gpu_ids)] if gpu_ids else None
            if train_expert_model(config_file, expert_name, gpu_id):
                success_count += 1
            else:
                print(f"警告: {expert_name}专家训练失败")

        print(f"\n专家模型训练完成: {success_count}/{len(expert_configs)} 成功")

    # 第二阶段
    if args.stage in ['2', 'all']:
        print("\n" + "=" * 50)
        print("第二阶段训练")
        print("=" * 50)

        # 检查专家模型
        if not check_expert_models():
            print("警告: 部分专家模型未找到，但继续MoE训练")

        # 训练MoE模型
        gpu_id = gpu_ids[0] if gpu_ids else None
        if train_moe_model(gpu_id):
            print("MoE融合训练完成!")
        else:
            print("MoE融合训练失败")

    # 评估阶段
    if args.evaluate:
        evaluate_models()

    print("\n" + "=" * 50)
    print("训练流程完成!")
    print("使用 'python demo.py' 来测试训练好的模型")
    print("=" * 50)


if __name__ == '__main__':
    main()
