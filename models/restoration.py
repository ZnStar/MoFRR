import torch
import numpy as np
import utils
import os
import torch.nn.functional as F


def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


class DiffusiveRestoration:
    def __init__(self, diffusion, args, config):
        super(DiffusiveRestoration, self).__init__()
        self.args = args
        self.config = config
        self.diffusion = diffusion

        if os.path.isfile(args.resume):
            self.diffusion.load_ddm_ckpt(args.resume, ema=True)
            self.diffusion.model.eval()
        else:
            print('Pre-trained diffusion model path is missing!')

    def restore(self, test_loader):
        # image_folder = os.path.join(self.args.image_folder, self.config.data.val_dataset)
        image_folder = os.path.join(self.args.image_folder, self.args.test_name)
        with torch.no_grad():
            for i, (x, y, id) in enumerate(test_loader):
                if i >= self.args.num_test:  # only apply our model to opt.num_test images.
                    print("already evaluate num:",self.args.num_test)
                    break
                x_cond = x[:, :3, :, :].to(self.diffusion.device) # 控制只读取美白图像
                b, c, h, w = x_cond.shape
                img_h_32 = int(32 * np.ceil(h / 32.0))
                img_w_32 = int(32 * np.ceil(w / 32.0))
                x_cond = F.pad(x_cond, (0, img_w_32 - w, 0, img_h_32 - h), 'reflect')
                x_output = self.diffusive_restoration(x_cond,y)
                x_output = x_output[:, :, :h, :w] 
                utils.logging.save_image(x_output, os.path.join(image_folder,f"{id[0]}")) #保存名字id_标签
                print(f"processing image {id[0]}")

    def diffusive_restoration(self, x_cond,y):
        x_output = self.diffusion.model(x_cond,y) 
        return x_output["pred_x"]

