#!/bin/bash

# MoFRR 模型评估脚本
# 用于评估训练好的专家模型和MoE融合模型

echo "MoFRR 模型评估脚本"
echo "==================="

# 评估专家模型
echo "评估专家模型..."

echo "评估眼睛专家模型..."
python evaluate.py --config configs/RetouchingOne_eye_uncond.yml --test_name eye_expert

echo "评估脸部专家模型..."
python evaluate.py --config configs/RetouchingOne_face_uncond.yml --test_name face_expert

echo "评估美白专家模型..."
python evaluate.py --config configs/RetouchingOne_white_uncond.yml --test_name white_expert

echo "评估平滑专家模型..."
python evaluate.py --config configs/RetouchingOne_smooth_uncond.yml --test_name smooth_expert

# 评估MoE模型
echo "评估MoE融合模型..."
python MoE_test.py --config MoE/RetouchingOne_MoE_uncond.yml --test_name MoE_fusion

echo "评估完成！结果保存在 results/ 目录中"


