import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import utils
from models.unet import DiffusionUNet
from models.wavelet import DWT, IWT
from pytorch_msssim import ssim
from skimage.metrics import peak_signal_noise_ratio as PSNR
# from models.mods import HFRM
from models.mods import HF_Crossattention
from models.DE import *
from torchvision import transforms
import random
from pytorch_grad_cam import GradCAM
import torchvision.models as models
# from pytorch_grad_cam.utils.image import scale_cam_image

def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class EMAHelper(object):
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Net(nn.Module):
    def __init__(self, args, config):
        super(Net, self).__init__()

        self.args = args
        self.config = config
        self.device = config.device

        # ########### uncond测试 需要################
        self.high_enhance0 = HF_Crossattention(in_channels=3, out_channels=64)
        self.high_enhance1 = HF_Crossattention(in_channels=3, out_channels=64)
        #####################################
        # if self.config.model.HF == 'HF_Crossattention':
        #     self.high_enhance0 = HF_Crossattention(in_channels=3, out_channels=64)
        #     self.high_enhance1 = HF_Crossattention(in_channels=3, out_channels=64)
        # elif self.config.model.HF == :
        #     # 待修改!!!
        #     self.high_enhance = iAFF()
    
        
        # if self.config.model.LF:
        #     # 输入xt,x_cond,y_emb,t，主要在y_emb做文章
        #     if self.config.model.DE_input == "GradCAM":
        #         inp_channels = 2 * self.config.data.channels + 2 +1 # xt,x_cond,info,t
        #         self.model_path = self.config.model.gradcam_modelpath
        #     elif self.config.model.DE_input == "degree":
        #         inp_channels = 2 * self.config.data.channels + 1 +1
        #     else:
        #         print("Error! please check the DE_input mode!")
        #     self.res_pred = globals()[self.config.model.DE_model](inp_channels=inp_channels) # 需要调整！！！！增加Respre模块,Respred_SimpleNet()等
        # self.res_pred = Respred_SimpleNet() # 之前没调整，res_simple得用这个
        
        self.Unet = DiffusionUNet(config)

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )

        self.betas = torch.from_numpy(betas).float()
        self.num_timesteps = self.betas.shape[0]

    @staticmethod
    def compute_alpha(beta, t):
        beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a

    def sample_training(self, x_cond, b, degree_emb,eta=0.,info_emb = None): # 增加标签y
        skip = self.config.diffusion.num_diffusion_timesteps // self.args.sampling_timesteps
        seq = range(0, self.config.diffusion.num_diffusion_timesteps, skip)
        n, c, h, w = x_cond.shape
        seq_next = [-1] + list(seq[:-1])
        x = torch.randn(n, c, h, w, device=self.device)
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(self.device)
            next_t = (torch.ones(n) * j).to(self.device)
            at = self.compute_alpha(b, t.long())
            at_next = self.compute_alpha(b, next_t.long())
            xt = xs[-1].to(self.device)
            
            ######################################
            if self.config.model.LF:  # 测试中改用预测的残差值
                residual = self.res_pred(xt,x_cond,info_emb,t)
                y_emb = torch.cat([degree_emb,residual], dim=1) # 3
            else:
                y_emb = degree_emb # 1
            ######################################
            # print(x_cond.dtype,xt.dtype,y_emb.dtype)
            et = self.Unet(torch.cat([x_cond, xt, y_emb], dim=1), t) # 直接把低频和xt合并之后作为条件输给Unet,[4,3*2+1,,h,w]
            # et = self.Unet(torch.cat([x_cond, xt, torch.full((xt.shape[0], 1, xt.shape[2], xt.shape[3]), y.item(), device=self.device)], dim=1), t) # 直接把低频和xt合并之后作为条件输给Unet,[4,3*2+1,,h,w]
            
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to(self.device))

        return xs[-1]

    # def pred_CAM(self,image_tensor,model_path):
    #     grad_enabled = torch.is_grad_enabled()
    
    #     # 在进入函数时，始终开启梯度追踪
    #     torch.set_grad_enabled(True)
    #     def load_model(model,model_path):
    #         """{'step':step,'state_dict':detection_model.state_dict(),
    #                                         'optimizer': optimizer.state_dict(),
    #                                         'scheduler':scheduler.state_dict(),
    #                                         'params': args,
    #                                         'config': config,
    #                                         'val_acc': val_acc}
    #         """
    #         checkpoint = torch.load(model_path)
    #         model.load_state_dict(checkpoint['state_dict'])
    #         model.eval()  # Set the model to evaluation mode
    #         return model
    #     test_transforms = transforms.Compose([
    #         transforms.Resize(512),
    #         # transforms.CenterCrop(224),
    #         # transforms.ToTensor(),
    #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #     ]) 
    #     detection_model = models.resnet50(pretrained=True)
    #     # 获取全连接层的输入特征数，替换全连接层
    #     num_ftrs = detection_model.fc.in_features
    #     num_classes = 5
    #     detection_model.fc = nn.Linear(num_ftrs, num_classes)
    #     detection_model.to(self.device)
    #     model = load_model(detection_model,model_path)
    #     model.eval()
    #     # 选择目标层，CAM对象
    #     target_layer = [model.layer4[-1]]
    #     cam = GradCAM(model=model, target_layers=target_layer)

    #     # 计算cam和标签
    #     image_tensor = test_transforms(image_tensor)
    #     grayscale_cam = cam(input_tensor=image_tensor)
    #     # grayscale_cam = scale_cam_image(grayscale_cam,(1024,1024)) # 缩放
    #     # with torch.no_grad():
    #     #     output = model(image_tensor)
    #     #     probs = F.softmax(output,dim=1)
    #     #     y_pred_soft = torch.sum(probs * degrees, dim=1)  # 软标签
    #     # 函数结束后，根据外部环境需要恢复梯度追踪状态
    #     if not grad_enabled:
    #         torch.set_grad_enabled(False)
    #     return torch.tensor(grayscale_cam).detach().unsqueeze(1)
    
    def forward(self, x, y=torch.tensor([-1.])): 
        data_dict = {}
        dwt, idwt = DWT(), IWT()
        y = y.to(self.device)
        input_img = x[:, :3, :, :].to(self.device)
        # input_degree = y
        
        n, c, h, w = input_img.shape
        input_img_norm = data_transform(input_img)
        input_dwt = dwt(input_img_norm)

        input_LL, input_high0 = input_dwt[:n, ...], input_dwt[n:, ...]
        # if self.args.removeHFRM is False: # 去掉HFRM不影响效果
        #     input_high0 = self.high_enhance0(input_high0)
        
        input_LL_dwt = dwt(input_LL)
        input_LL_LL, input_high1 = input_LL_dwt[:n, ...], input_LL_dwt[n:, ...]
        
        # if self.args.removeHFRM is False: # 去掉HFRM不影响效果
        #     input_high1 = self.high_enhance1(input_high1)

        b = self.betas.to(self.device)

        t = torch.randint(low=0, high=self.num_timesteps, size=(input_LL_LL.shape[0] // 2 + 1,))
        t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:input_LL_LL.shape[0]].to(self.device)
        a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)

        e = torch.randn_like(input_LL_LL)
        
        degree_emb = y.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(-1,1,input_LL_LL.shape[2],input_LL_LL.shape[3]) # 只是单纯将预测的程度扩充，（b,1,h,w)
        
        if self.training:
            gt_img_norm = data_transform(x[:, 3:, :, :]).to(self.device)
            gt_dwt = dwt(gt_img_norm)
            gt_LL, gt_high0 = gt_dwt[:n, ...], gt_dwt[n:, ...]

            gt_LL_dwt = dwt(gt_LL)
            gt_LL_LL, gt_high1 = gt_LL_dwt[:n, ...], gt_LL_dwt[n:, ...]

            x = gt_LL_LL * a.sqrt() + e * (1.0 - a).sqrt()  # 这个x看起来是gt正向加噪的xt
            
            
            ####################### res控制一下加入残差 只改y_emb#########################            
            # if self.config.model.LF:  # 这部分是用来算loss的，在sample_training里面是主体
            #     if self.config.model.DE_input == "GradCAM":
            #         # print(input_img.shape) # torch.Size([16, 3, 256, 256])
            #         gradcam = self.pred_CAM(input_img,self.model_path).to(self.device)
            #         gradcam = F.interpolate(gradcam, size=(degree_emb.shape[-1],degree_emb.shape[-1]), mode='bilinear', align_corners=False) # 统一尺寸
            #         # print(degree_emb.shape) 
            #         # print(gradcam.shape)
            #         info_emb = torch.cat([degree_emb,gradcam], dim=1) # 统一torch.Size([16, 1, 64, 64])
            #     elif self.config.model.DE_input == "degree":
            #         info_emb = degree_emb
            #     y_res= self.res_pred(x,input_LL_LL,info_emb,t) # 用xt_gt,input，y（之后修改y），t作为输入，输出像素级失真预测
            #     y_emb = torch.cat([degree_emb,y_res], dim=1)
            #     data_dict["res_pred"] = y_res
            #     data_dict["res_true"] = gt_LL_LL-input_LL_LL 
            # else:
            y_emb = degree_emb
            info_emb = None
            noise_output = self.Unet(torch.cat([input_LL_LL, x, y_emb], dim=1), t.float()) # 在Unet中需要添加y,gt    
            
            denoise_LL_LL = self.sample_training(input_LL_LL, b,degree_emb,info_emb=info_emb) # 输出是迭代预测到的x0，在Unet中需要添加y，这里不应该加低频，而是在迭代内部加
            ##############################################################################
            
            ################借助xt修改高频######################
            # if self.config.model.HF == 'HF_Crossattention':
            #     input_high0 = self.high_enhance0(input_high0,denoise_LL_LL)
            #     input_high1 = self.high_enhance1(input_high1,denoise_LL_LL)
            # elif self.config.model.HF == 'iAFF':
            #     input_high0, input_high1 = self.high_enhance(input_high0,input_high1,denoise_LL_LL)
            ###################################################
            pred_LL = idwt(torch.cat((denoise_LL_LL, input_high1), dim=0))

            pred_x = idwt(torch.cat((pred_LL, input_high0), dim=0))
            pred_x = inverse_data_transform(pred_x)
            
            data_dict["input_high0"] = input_high0
            data_dict["input_high1"] = input_high1
            data_dict["gt_high0"] = gt_high0
            data_dict["gt_high1"] = gt_high1
            data_dict["pred_LL"] = pred_LL
            data_dict["gt_LL"] = gt_LL
            data_dict["noise_output"] = noise_output
            data_dict["pred_x"] = pred_x
            data_dict["e"] = e

        else:
            ############# LF #############################
            # if self.config.model.LF:  
            #     if self.config.model.DE_input == "GradCAM":
            #         # print(input_img.shape) #torch.Size([1, 3, 1024, 1024])
            #         gradcam = self.pred_CAM(input_img.detach(),self.model_path).to(self.device)
            #         gradcam = F.interpolate(gradcam, size=(degree_emb.shape[-1],degree_emb.shape[-1]), mode='bilinear', align_corners=False) # 统一尺寸
            #         info_emb = torch.cat([degree_emb,gradcam], dim=1)
            #         # print(info_emb.shape)
            #     elif self.config.model.DE_input == "degree":
            #         info_emb = degree_emb
            # else:
            info_emb=None
            denoise_LL_LL = self.sample_training(input_LL_LL, b,degree_emb,info_emb=info_emb)# 加y
            
            ################借助xt修改高频######################
            # if self.config.model.HF == 'HF_Crossattention':
            #     input_high0 = self.high_enhance0(input_high0,denoise_LL_LL)
            #     input_high1 = self.high_enhance1(input_high1,denoise_LL_LL)
            # elif self.config.model.HF == 'iAFF':
            #     input_high0, input_high1 = self.high_enhance(input_high0,input_high1,denoise_LL_LL)
            ###################################################
            pred_LL = idwt(torch.cat((denoise_LL_LL, input_high1), dim=0))
            pred_x = idwt(torch.cat((pred_LL, input_high0), dim=0))
            pred_x = inverse_data_transform(pred_x)

            data_dict["pred_x"] = pred_x

        return data_dict


class DenoisingDiffusion(object):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.device = config.device

        self.model = Net(args, config)
        self.model.to(self.device)
        self.model = torch.nn.DataParallel(self.model)

        self.ema_helper = EMAHelper()
        self.ema_helper.register(self.model)

        self.l2_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss()
        self.TV_loss = TVLoss()

        self.optimizer, self.scheduler = utils.optimize.get_optimizer(self.config, self.model.parameters())
        self.start_epoch, self.step = 0, 0

    def load_ddm_ckpt(self, load_path, ema=False):
        checkpoint = utils.logging.load_checkpoint(load_path, None)
        # print(checkpoint['state_dict'].keys())
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.ema_helper.load_state_dict(checkpoint['ema_helper'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])        
        self.step = checkpoint['step']
        self.start_epoch = checkpoint['epoch']
        
        if ema:
            self.ema_helper.ema(self.model)
        # print("=> loaded checkpoint {} step {}".format(load_path, self.step))
        utils.logging.log("=> loaded checkpoint {} step {}".format(load_path, self.step),file_path=self.config.log_path)
    def train(self, DATASET):
        cudnn.benchmark = True
        train_loader, val_loader = DATASET.get_loaders()

        if os.path.isfile(self.args.resume):
            self.load_ddm_ckpt(self.args.resume)

        best_val_PSNR = 0
        for epoch in range(self.start_epoch, self.config.training.n_epochs):
            # print('epoch: ', epoch)
            utils.logging.log(f"epoch:{epoch}",file_path=self.config.log_path)
            data_start = time.time()
            data_time = 0
            for i, (x, y, name) in enumerate(train_loader): # y之前是图像id且没用到，改成程度作为条件
                if i >= len(train_loader)-1: 
                    break
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                data_time += time.time() - data_start
                self.model.train()
                self.step += 1

                x = x.to(self.device)

                # output = self.model(x)
                output = self.model(x,y)
                
                noise_loss, photo_loss, frequency_loss,respred_loss = self.estimation_loss(x, output) # 增加了respred_loss
                loss = noise_loss + photo_loss + frequency_loss + respred_loss
                
                if self.step % 10 == 0:
                    # print("step:{}, lr:{:.6f}, noise_loss:{:.4f}, photo_loss:{:.4f}, "
                    #       "frequency_loss:{:.4f}".format(self.step, self.scheduler.get_last_lr()[0],
                    #                                      noise_loss.item(), photo_loss.item(),
                    #                                      frequency_loss.item()))
                    log_message = "step:{}, lr:{:.6f}, noise_loss:{:.4f}, photo_loss:{:.4f}, frequency_loss:{:.4f},respred_loss:{:.4f}".format(
                            self.step, self.scheduler.get_last_lr()[0], noise_loss.item(), photo_loss.item(), frequency_loss.item(), respred_loss.item())

                    # 使用函数将消息写入文件
                    utils.logging.log(log_message,file_path=self.config.log_path)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.ema_helper.update(self.model)
                data_start = time.time()

                if self.step % self.config.training.validation_freq == 0 and self.step != 0:
                    self.model.eval()
                    val_PSNR = self.sample_validation_patches(val_loader, self.step, self.config.training.num_val) # 随机200/50验证
                    if val_PSNR > best_val_PSNR: # 如果是最优的，保存模型
                        best_val_PSNR = val_PSNR
                        utils.logging.save_checkpoint({'step': self.step, 'epoch': epoch + 1,
                                                   'state_dict': self.model.state_dict(),
                                                   'optimizer': self.optimizer.state_dict(),
                                                   'scheduler': self.scheduler.state_dict(),
                                                   'ema_helper': self.ema_helper.state_dict(),
                                                   'params': self.args,
                                                   'config': self.config,
                                                   'PSNR': best_val_PSNR},
                                                  filename=os.path.join(self.config.data.ckpt_dir, 'model_best_'+self.config.data.train_dataset+'_step_'+str(self.step)))

                    utils.logging.save_checkpoint({'step': self.step, 'epoch': epoch + 1,
                                                   'state_dict': self.model.state_dict(),
                                                   'optimizer': self.optimizer.state_dict(),
                                                   'scheduler': self.scheduler.state_dict(),
                                                   'ema_helper': self.ema_helper.state_dict(),
                                                   'params': self.args,
                                                   'config': self.config},
                                                  filename=os.path.join(self.config.data.ckpt_dir, 'model_latest'+self.config.data.train_dataset))
                        
            self.scheduler.step()

    def estimation_loss(self, x, output):

        input_high0, input_high1, gt_high0, gt_high1 = output["input_high0"], output["input_high1"],\
                                                       output["gt_high0"], output["gt_high1"]

        pred_LL, gt_LL, pred_x, noise_output, e = output["pred_LL"], output["gt_LL"], output["pred_x"],\
                                                  output["noise_output"], output["e"]

        gt_img = x[:, 3:, :, :].to(self.device)
        # =============noise loss==================
        noise_loss = self.l2_loss(noise_output, e)
        # =============frequency loss==================
        if self.config.model.HF:
            frequency_loss = 0.1 * (self.l2_loss(input_high0, gt_high0) +
                                    self.l2_loss(input_high1, gt_high1) +
                                    self.l2_loss(pred_LL, gt_LL)) +\
                            0.01 * (self.TV_loss(input_high0) +
                                    self.TV_loss(input_high1) +
                                    self.TV_loss(pred_LL))
        else:
            frequency_loss = 0.1 * self.l2_loss(pred_LL, gt_LL) + 0.01 * self.TV_loss(pred_LL)

        # =============photo loss==================
        content_loss = self.l1_loss(pred_x, gt_img)
        ssim_loss = 1 - ssim(pred_x, gt_img, data_range=1.0).to(self.device)

        photo_loss = content_loss + ssim_loss
        ############## respred loss ###############
        if self.config.model.LF:
            respred_loss = 0.1 * self.l2_loss(output["res_pred"],output["res_true"])
            # print(f'noise_loss:{noise_loss}, photo_loss:{photo_loss}, frequency_loss:{frequency_loss},respred_loss:{respred_loss}')
        else:
            respred_loss = torch.zeros_like(noise_loss)
        ###########################################
        return noise_loss, photo_loss, frequency_loss, respred_loss

    def sample_validation_patches(self, val_loader, step, num_samples): # 需要改一下，太费时间，也不清楚效果怎么样，需要有效输出PSNR比较
        image_folder = os.path.join(self.args.image_folder, self.config.data.type +self.config.data.train_dataset + str(self.config.data.patch_size))
        self.model.eval()
        
        # 随机选取部分图像
        # total_samples = len(val_loader.dataset)-1 # 是样本总数,修改batchsize后迭代次数少很多所以总报错
        total_samples = len(val_loader.dataset)//val_loader.batch_size -1
        sample_indices = random.sample(range(total_samples), min(num_samples, total_samples)) # 最后一个未必全，不用
        positive_percentage = 0
        with torch.no_grad():
            # print(f"Processing a single batch of validation images at step: {step}")
            utils.logging.log(f"Processing a single batch of validation images at step: {step}",file_path=self.config.log_path)
            PSNR_list = []
            PSNR_pred_list = []
            for i, (x, y, name) in enumerate(val_loader):
                if i not in sample_indices:
                    continue
                b, _, img_h, img_w = x.shape
                img_h_32 = int(32 * np.ceil(img_h / 32.0)) # 调整为 32 的倍数
                img_w_32 = int(32 * np.ceil(img_w / 32.0))
                x = F.pad(x, (0, img_w_32 - img_w, 0, img_h_32 - img_h), 'reflect')

                # print(x.shape,y.shape) #torch.Size([1, 6, 1024, 1024])
                out = self.model(x.to(self.device),y)
                pred_x = out["pred_x"]
                pred_x = pred_x[:, :, :img_h, :img_w].to('cpu')
                
                # 增加测试PSNR的大小
                gt_x = x[:, 3:, :, :]
                input_x = x[:, :3, :, :]
                # PSNR_pred = utils.calculate_psnr(pred_x,gt_x)
                # PSNR_pred_list.append(PSNR_pred)
                # PSNR_ori = utils.calculate_psnr(input_x,gt_x)
                # # print(PSNR_ori)
                # PSNR_list.append(PSNR_pred-PSNR_ori)
                # print(gt_x.shape,input_x.shape,pred_x.shape)
                for ind in range(len(pred_x)):
                    PSNR_pred = PSNR(pred_x[ind].cpu().detach().numpy().transpose(1, 2, 0),gt_x[ind].cpu().detach().numpy().transpose(1, 2, 0))
                    PSNR_pred_list.append(PSNR_pred)
                    PSNR_ori = PSNR(input_x[ind].cpu().detach().numpy().transpose(1, 2, 0),gt_x[ind].cpu().detach().numpy().transpose(1, 2, 0))
                    # print(PSNR_ori)
                    PSNR_list.append(PSNR_pred-PSNR_ori)
                positive_percentage = (sum(1 for x in PSNR_list if x > 0) / len(PSNR_list)) * 100
                
                utils.logging.save_image(pred_x, os.path.join(image_folder, str(step),f"y{y[0]}_{name[0]}"))

            utils.logging.log(f"validation num:{len(PSNR_list)},average PSNR: {np.mean(PSNR_pred_list)},average PSNR lift: {np.mean(PSNR_list)}, positive percentage:{positive_percentage}%",file_path=self.config.log_path)
        return np.mean(PSNR_list)