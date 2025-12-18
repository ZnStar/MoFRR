#!/bin/bash

# MoE模型训练和测试脚本

# 训练MoE模型
# CUDA_VISIBLE_DEVICES=0,1,2,3 python MoE_train.py --config MoE/RetouchingOne_MoE_uncond.yml

# 测试MoE模型
CUDA_VISIBLE_DEVICES=0,1 python MoE_test.py --config MoE/RetouchingOne_MoE_test50.yml --test_name MoE_test50 --resume ckpt/MoE/uncond/model_best_MoE_uncond.pth.tar

# 计算FLOPs
CUDA_VISIBLE_DEVICES=0 python test_FLOP.py --config MoE/RetouchingOne_MoE_uncond.yml --test_name MoE_uncond --resume ckpt/MoE/uncond/model_best_MoE_uncond.pth.tar