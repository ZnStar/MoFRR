########## 训练
# CUDA_VISIBLE_DEVICES=1,3,0 python MoE_train.py --config /newdata/DiffRR/MoE/configs/RetouchingOne_MoE_uncond.yml
########## 测试(restore要对应看)
# CUDA_VISIBLE_DEVICES=0,1 python MoE_test.py --config /newdata/DiffRR/MoE/configs/RetouchingOne_MoE_uncond.yml --test_name MoE_uncond --resume /newdata/DiffRR/MoE/ckpt/uncond/model_best_MoE_uncond_step185050_PSNR6.58.pth.tar
# CUDA_VISIBLE_DEVICES=0,1 python MoE_test.py --config /newdata/DiffRR/MoE/configs/RetouchingOne_MoE_uncond.yml --test_name MoE_uncond --resume /newdata/DiffRR/MoE/ckpt/uncond/model_best_MoE_uncond_step185050_PSNR6.58.pth.tar
CUDA_VISIBLE_DEVICES=0,1 python MoE_test.py --config /newdata/DiffRR/MoE/configs/RetouchingOne_MoE_test50.yml --test_name MoE_test50 --resume /newdata/DiffRR/MoE/ckpt/uncond/model_best_MoE_uncond_step185050_PSNR6.58.pth.tar

CUDA_VISIBLE_DEVICES=0 python test_FLOP.py --config /newdata/DiffRR/MoE/configs/RetouchingOne_MoE_uncond.yml --test_name MoE_uncond --resume /newdata/DiffRR/MoE/ckpt/uncond/model_best_MoE_uncond_step185050_PSNR6.58.pth.tar