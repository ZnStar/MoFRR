import argparse
import os
import random
import socket
import yaml
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import models
import datasets
import utils
from models import DenoisingDiffusion, DiffusiveRestoration


def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Evaluate Wavelet-Based Diffusion Model')
    ################## 需要修改 ############################
    parser.add_argument("--config", default='portrait_smooth_uncond.yml', type=str,
                        help="Path to the config file")
    parser.add_argument("--test_name", default='portrait_smooth_uncond90', type=str,
                        help="test dataset name")
    parser.add_argument('--resume', default='ckpt/model_latestsmooth_all_uncond.pth.tar', type=str,
                        help='Path for the diffusion model checkpoint to load for evaluation')
    ### 单独测试文件夹
    parser.add_argument("--image_folder", default='results/test', type=str,
                        help="Location to save restored images")
    parser.add_argument('--num_test', default=5000, type=int, 
                        help='number of test dataset') # 随机性太大
    parser.add_argument("--test_folder", default=None, type=str,
                        help="Path to the test image folder") # 用单独的文件夹测试才用到,平常默认None！
    ########################################################
    
    # parser.add_argument('--resume', default='ckpt/model.pth.tar', type=str,
    #                     help='Path for the diffusion model checkpoint to load for evaluation')
    parser.add_argument("--sampling_timesteps", type=int, default=10,
                        help="Number of implicit sampling steps")
    parser.add_argument('--seed', default=123, type=int, metavar='N',
                        help='Seed for initializing training (default: 230)')
    args = parser.parse_args()

    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()

    # setup device to run
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: {}".format(device))
    config.device = device

    if torch.cuda.is_available():
        print('Note: Currently supports evaluations (restoration) when run only on a single GPU!')

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    # data loading
    config.data.patch_size=config.data.resolution # 控制结果不是块,而是特定分辨率eg1024
    print("=> using dataset '{}'".format(args.test_folder))
    DATASET = datasets.__dict__[config.data.type](config)
    # _, val_loader = DATASET.get_loaders(parse_patches=False) #奇怪的参数
    test_loader = DATASET.get_loaders(test=True,test_folder=args.test_folder)  # 单一测试文件夹
        
    
    
    # create model
    print("=> creating denoising-diffusion model")
    diffusion = DenoisingDiffusion(args, config)
    model = DiffusiveRestoration(diffusion, args, config)
    model.restore(test_loader)


if __name__ == '__main__':
    main()
