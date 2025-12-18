import argparse
import os
import random
import yaml
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import numpy as np
import torchvision
import datasets
import utils
import time
# from models import MoE_model
from MoE import RetouchingMulti,MoE_model

def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Evaluate Wavelet-Based Diffusion Model')
    ################## 需要修改 ############################
    parser.add_argument("--config", type=str, help="Path to the config file")
    parser.add_argument("--image_folder", default='/newdata/DiffRR/MoE/results/', type=str,
                        help="Location to save restored validation image patches")
    # parser.add_argument("--test_name", default='portrait_smooth_uncond90', type=str,
    #                     help="test dataset name")
    # parser.add_argument('--resume', default='ckpt/model_latestsmooth_all_uncond.pth.tar', type=str,
    #                     help='Path for the diffusion model checkpoint to load for evaluation')
    # ### 单独测试文件夹
    # parser.add_argument("--image_folder", default='results/test', type=str,
    #                     help="Location to save restored images")
    # parser.add_argument('--num_test', default=5000, type=int, 
    #                     help='number of test dataset') # 随机性太大
    # parser.add_argument("--test_folder", default=None, type=str,
    #                     help="Path to the test image folder") # 用单独的文件夹测试才用到,平常默认None！
    # ########################################################
    
    # # parser.add_argument('--resume', default='ckpt/model.pth.tar', type=str,
    # #                     help='Path for the diffusion model checkpoint to load for evaluation')
    # parser.add_argument("--sampling_timesteps", type=int, default=10,
    #                     help="Number of implicit sampling steps")
    parser.add_argument('--seed', default=123, type=int, metavar='N',
                        help='Seed for initializing training')
    args = parser.parse_args()

    with open(os.path.join("/newdata/DiffRR/MoE/configs", args.config), "r") as f:
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

def seed_all(set_seed = 123):
    torch.manual_seed(set_seed)
    np.random.seed(set_seed)
    random.seed(set_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(set_seed)
    torch.backends.cudnn.benchmark = True   

if __name__ == '__main__':
    # configs and args: !!!!!!!!!!!!!!!!!!!!!
    args, config = parse_args_and_config()
    
    # devices
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    utils.logging.log("Using device: {}".format(device),file_path=config.log_path)
    config.device = device

    if torch.cuda.is_available():
        print('Note: Currently supports evaluations (restoration) when run only on a single GPU!')

    # set random seed
    seed_all(set_seed = args.seed)
    
    # data loading !!!!!!!!!!!!!!!!!!!!!
    utils.logging.log("=> using dataset '{}'".format(config.name),file_path=config.log_path)
    DATASET = RetouchingMulti(config)
    # datasets.__dict__[config.data.type](config)
    
    # MoE
    utils.logging.log("=> creating denoising-diffusion model...",file_path=config.log_path)
    model = MoE_model(args, config)
    model.train(DATASET)