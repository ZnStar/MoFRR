# MoFRR: Mixture of Diffusion Models for Face Retouching Restoration

Paper Link: [ICCV 2025 Paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Liu_MoFRR_Mixture_of_Diffusion_Models_for_Face_Retouching_Restoration_ICCV_2025_paper.pdf)


## Dataset

Please download the [Application](https://fdmas.github.io/Application_RetouchingFFHQ_new.pdf), fill it out according to the instructions, and send it to the given email address to obtain the dataset.

## Environment Requirements

```bash
pip install -r requirements.txt
```

Main Dependencies:
- PyTorch >= 1.8.1
- CUDA >= 10.2
- OpenCV
- NumPy

## Quick Start

### Training

Use the provided `train_experts.py` script for training:

```bash
# Train all stages
python train_experts.py

# Train stage 1
python train_experts.py --stage 1

# Train stage 2
python train_experts.py --stage 2
```

### Performance Evaluation


```bash
# Evaluate expert models
python evaluate.py --config configs/RetouchingOne_[expert].yml

# Evaluate MoFRR
python MoE_test.py --config MoE/RetouchingOne_MoE_uncond.yml
```


## Citation

If you use this project in your research, please cite our paper:

```
@inproceedings{liu2025mofrr,
  title={MoFRR: Mixture of Diffusion Models for Face Retouching Restoration},
  author={Liu, Jiaxin and Ying, Qichao and Qian,Zhenxing and Li, Sheng and Zhang, Runqi and Liu, Jian and Zhang, Xinpeng},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2025}
}
```

