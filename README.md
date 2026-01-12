# MoFRR: Mixture of Diffusion Models for Face Retouching Restoration

## 项目概述

论文链接: [ICCV 2025 Paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Liu_MoFRR_Mixture_of_Diffusion_Models_for_Face_Retouching_Restoration_ICCV_2025_paper.pdf)


## 数据获取

请下载[数据使用申请表](https://fdmas.github.io/Application_RetouchingFFHQ_new.pdf)，并按照表格要求填写并发送至相应邮箱获取数据集。

## 环境要求

```bash
pip install -r requirements.txt
```

主要依赖：
- PyTorch >= 1.8.1
- CUDA >= 10.2
- OpenCV
- NumPy
- 其他依赖见 requirements.txt


## 快速开始

### 训练

使用提供的 `train_experts.py` 脚本进行自动化训练：

```bash
# 训练所有阶段（专家模型 + MoE融合）
python train_experts.py

# 仅训练专家模型
python train_experts.py --stage 1

# 仅训练MoE融合模型
python train_experts.py --stage 2
```

### 性能评估


```bash
# 专家模型评估
python evaluate.py --config configs/RetouchingOne_[expert].yml

# MoE融合模型评估
python MoE_test.py --config MoE/RetouchingOne_MoE_uncond.yml
```


## 引用

如果您在研究中使用了本项目，请引用我们的论文：

```
@inproceedings{liu2025mofrr,
  title={MoFRR: Mixture of Diffusion Models for Face Retouching Restoration},
  author={Liu, Shuaicheng and others},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2025}
}
```

## 许可证

本项目采用MIT许可证。
