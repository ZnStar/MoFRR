# 注意输出的y是wsfe的程度列表！！！
# name里面有无wsfe决定是不是调用！！！
import os
import torch
import torch.utils.data
import PIL
from PIL import Image
from datasets.data_augment import PairCompose, PairRandomCrop, PairToTensor
import random
import pandas as pd

class RetouchingMulti():
    def __init__(self, config):
        self.config = config

    def combine_datasets(self, split):
        lq_data_dirs = [os.path.join(self.config.data.lq_data_dir,folder, split) for folder in self.config.data.lq_data_type]
        hq_data_dir = os.path.join(self.config.data.hq_data_dir)
        is_train = False
        if self.config.data.conditional:
            y_info = self.config.data.condition_type
            info_df = pd.read_csv(self.config.data.info_path)
            info_df.set_index('file_path', inplace=True)
        else:
            y_info = False
            info_df = None
        
        patch_size=self.config.data.resolution # 控制结果大小
        if split =='test':
            # hq_data_dir = lq_data_dirs[0] # 反正测试不会用到
            pass
        elif split =='train':
            is_train = True
            patch_size=self.config.data.patch_size

        combined_dataset = []
        for lq_dir in lq_data_dirs:
            dataset = Paired_Dataset(lq_dir, hq_data_dir, patch_size=patch_size,train=is_train,y_info=y_info,info_df = info_df)
            combined_dataset.append(dataset)

        return torch.utils.data.ConcatDataset(combined_dataset)
    
    def get_loaders(self,test=False,test_folder=None):
        
        
        if test:
            if test_folder:
                # 图片的文件夹
                test_dataset = Paired_Dataset(test_folder,test_folder,
                                            patch_size=self.config.data.resolution,train=False) # 用整张图
                test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True,
                                                    num_workers=self.config.data.num_workers,
                                                    pin_memory=True)
            else:
                # 不同程度都算
                test_dataset = self.combine_datasets('test')
                test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True,
                                                    num_workers=self.config.data.num_workers,
                                                    pin_memory=True)
            return test_loader
        

        train_dataset = self.combine_datasets('train')
        val_dataset = self.combine_datasets('val')
        # print("patch大小！！！",self.config.data.patch_size) # 默认切成256进行，那其实是先切片再小波了，diffusion大小是64的？

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.training.batch_size,
                                                   shuffle=True, num_workers=self.config.data.num_workers,
                                                   pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                                 num_workers=self.config.data.num_workers,
                                                 pin_memory=True)

        return train_loader, val_loader
    
class Paired_Dataset(torch.utils.data.Dataset):
    def __init__(self, lq_dir,hq_dir, patch_size, train=True,y_info=False,info_df=None):
        super().__init__()

        self.lq_dir = lq_dir
        self.hq_dir = hq_dir
        self.train = train
        self.name_list = os.listdir(lq_dir) # 文件名列表
        self.patch_size = patch_size
        self.y_info = y_info
        self.info_df = info_df
                
                
        if self.train:
            self.transforms = PairCompose([
                PairRandomCrop(self.patch_size),
                PairToTensor()                                                                                                                                                                                                                                                                
            ])
        else:
            self.transforms = PairCompose([
                PairToTensor()
            ])

    def get_images(self, index, p=1):
        file_name = self.name_list[index]
        name = file_name.split('_')[-1]
        # img_id = re.split('/', name)[-1][:-4] # 无后缀的图片名
        if self.y_info:
            if self.y_info == 'pred_info':
                my_info = float(self.info_df.at[os.path.join(self.lq_dir,file_name), 'pred_degree']) # 直接读取预测的类型
            elif self.y_info == 'gt':
                my_info = float(self.lq_dir.split("/")[-1]) # 文件夹路径/newdata/RetouchingOne/EyeEnlarging/train/2
            else:
                print("Please check your info type!!! Now uncond")
                my_info = -1
        else:
            my_info = -1 
        lq_img = Image.open(os.path.join(self.lq_dir, file_name))
        hq_img = Image.open(os.path.join(self.hq_dir, name))
        lq_img, hq_img = self.transforms(lq_img, hq_img)

        return torch.cat([lq_img, hq_img], dim=0), torch.tensor(my_info, dtype=torch.float32), file_name #增加了辅助信息，但id在保存还需要用到

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.name_list)
    
    
# train test val都要
class RetouchingMulti_all():
    def __init__(self, config):
        self.config = config

    def combine_datasets(self, splits):
        lq_data_dirs = [os.path.join(self.config.data.lq_data_dir,folder, split) for folder in self.config.data.lq_data_type for split in splits] #!
        hq_data_dir = os.path.join(self.config.data.hq_data_dir)
        is_train = False
        # if self.config.data.conditional:
        #     y_info = self.config.data.condition_type
        #     info_df = pd.read_csv(self.config.data.info_path)
        #     info_df.set_index('file_path', inplace=True)
        # else:
        y_info = False
        info_df = None
        
        patch_size=self.config.data.resolution # 控制结果大小
        # if split =='test':
        #     # hq_data_dir = lq_data_dirs[0] # 反正测试不会用到
        #     pass
        # elif split =='train':
        #     is_train = True
        #     patch_size=self.config.data.patch_size

        combined_dataset = []
        for lq_dir in lq_data_dirs:
            dataset = Paired_Dataset(lq_dir, hq_data_dir, patch_size=patch_size,train=is_train,y_info=y_info,info_df = info_df)
            combined_dataset.append(dataset)

        return torch.utils.data.ConcatDataset(combined_dataset)
    
    def get_loaders(self,test=False,test_folder=None):
        
        
        if test:
            if test_folder:
                # 图片的文件夹
                test_dataset = Paired_Dataset(test_folder,test_folder,
                                            patch_size=self.config.data.resolution,train=False) # 用整张图
                test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True,
                                                    num_workers=self.config.data.num_workers,
                                                    pin_memory=True)
            else:
                # 不同程度都算
                test_dataset = self.combine_datasets(['test'])
                test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True,
                                                    num_workers=self.config.data.num_workers,
                                                    pin_memory=True)
            return test_loader
        

        # train_dataset = self.combine_datasets('train')
        # val_dataset = self.combine_datasets('val')
        # # print("patch大小！！！",self.config.data.patch_size) # 默认切成256进行，那其实是先切片再小波了，diffusion大小是64的？

        # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.training.batch_size,
        #                                            shuffle=True, num_workers=self.config.data.num_workers,
        #                                            pin_memory=True)
        # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config.training.batch_size, shuffle=True,
        #                                          num_workers=self.config.data.num_workers,
        #                                          pin_memory=True)

        # return train_loader, val_loader