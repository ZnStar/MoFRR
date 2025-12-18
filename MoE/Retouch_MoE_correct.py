from models.ddm_uncond import *
import argparse
import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from pytorch_msssim import ssim
from skimage.metrics import peak_signal_noise_ratio as PSNR
import torch.optim as optim
from MoE.MoEUnet import *

def get_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return new_config

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


class MoE_model(object):
    def __init__(self,args,config):
        super().__init__()
        self.args = args
        self.config = config
        self.device = config.device
        self.MoEmodel = UNet(in_channels=3*(config.model.num_expert+1), out_channels = config.model.out_channel)
        self.MoEmodel.to(self.device)
        self.MoEmodel = torch.nn.DataParallel(self.MoEmodel, dim=0)
        
        self.config_white = get_config(config.model.white.config_path)
        self.config_eye = get_config(config.model.eye.config_path)
        self.config_face = get_config(config.model.face.config_path)
        self.config_smooth = get_config(config.model.smooth.config_path)
        self.config_white.device = config.device
        self.config_eye.device = config.device
        self.config_face.device = config.device
        self.config_smooth.device = config.device
        self.resume_path_white = config.model.white.resume_path
        self.resume_path_face = config.model.face.resume_path
        self.resume_path_eye = config.model.eye.resume_path
        self.resume_path_smooth = config.model.smooth.resume_path
        self.model_white = Net(args, self.config_white)
        self.model_white = torch.nn.DataParallel(self.model_white)
        self.model_face = Net(args, self.config_face)
        self.model_face = torch.nn.DataParallel(self.model_face)
        self.model_eye = Net(args, self.config_eye)
        self.model_eye= torch.nn.DataParallel(self.model_eye)
        self.model_smooth = Net(args, self.config_smooth)
        self.model_smooth = torch.nn.DataParallel(self.model_smooth)
        
        self.step = 0
        self.l2_loss = torch.nn.MSELoss()
        self.optimizer = optim.Adam(self.MoEmodel.parameters(), lr=config.optim.lr, weight_decay=config.optim.weight_decay,
                          betas=(0.9, 0.999), amsgrad=config.optim.amsgrad, eps=config.optim.eps)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=config.optim.step_size,
                                          gamma=config.optim.gamma, last_epoch=-1)
        self.ema_helper = EMAHelper()
        self.ema_helper.register(self.MoEmodel)
        
    def load_expert_ckpt(self):
        checkpoint_white = utils.logging.load_checkpoint(self.resume_path_white, None)
        self.model_white.load_state_dict(checkpoint_white['state_dict'], strict=True)
        for param in self.model_white.parameters():
            param.requires_grad = False
        self.model_white.eval()
        
        checkpoint_face = utils.logging.load_checkpoint(self.resume_path_face, None)
        self.model_face.load_state_dict(checkpoint_face['state_dict'], strict=True)
        for param in self.model_face.parameters():
            param.requires_grad = False
        self.model_face.eval()
        
        checkpoint_eye = utils.logging.load_checkpoint(self.resume_path_eye, None)
        self.model_eye.load_state_dict(checkpoint_eye['state_dict'], strict=True)
        for param in self.model_eye.parameters():
            param.requires_grad = False
        self.model_eye.eval()
        
        checkpoint_smooth = utils.logging.load_checkpoint(self.resume_path_smooth, None)
        self.model_smooth.load_state_dict(checkpoint_smooth['state_dict'], strict=True)
        for param in self.model_smooth.parameters():
            param.requires_grad = False
        self.model_smooth.eval()
        utils.logging.log("=> Experts loaded")
                
    def train(self, DATASET):
        cudnn.benchmark = True
        train_loader, val_loader = DATASET.get_loaders()
        
        # 读入并冻结模型
        utils.logging.log("=> loading Experts")
        self.load_expert_ckpt()
        
        best_val_PSNR = 0
        for epoch in range(self.config.training.n_epochs):
            utils.logging.log(f"epoch:{epoch}",file_path=self.config.log_path)
            for i, (x, y, name) in enumerate(train_loader):
                if i >= min(len(train_loader)-1,self.config.training.num_trainloader): 
                    break
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                x = x.to(self.device)
                self.step += 1
                # 读取输入图像，并且确保是32的整数倍
                x_gt = x[:,3:,:,:]
                x_cond = x[:, :3, :, :]
                b, c, h, w = x_cond.shape
                img_h_32 = int(32 * np.ceil(h / 32.0))
                img_w_32 = int(32 * np.ceil(w / 32.0))
                x_cond = F.pad(x_cond, (0, img_w_32 - w, 0, img_h_32 - h), 'reflect')
                
                MoE_input = x_cond
                if '0w' in name:
                    output_white = self.model_white(x,y[0])[:, :, :h, :w] - x_cond
                    MoE_input = torch.cat((MoE_input, output_white), dim=1)
                else:
                    MoE_input = torch.cat((MoE_input,torch.zeros_like(x_cond)), dim=1)
                if '0s' in name:
                    output_smooth = self.model_smooth(x,y[1])[:, :, :h, :w] - x_cond
                    MoE_input = torch.cat((MoE_input, output_smooth), dim=1)
                else:
                    MoE_input = torch.cat((MoE_input,torch.zeros_like(x_cond)), dim=1)
                if '0f' in name:
                    output_face = self.model_face(x,y[2])[:, :, :h, :w] - x_cond
                    MoE_input = torch.cat((MoE_input, output_face), dim=1)
                else:
                    MoE_input = torch.cat((MoE_input,torch.zeros_like(x_cond)), dim=1)
                if '0e' in name:
                    output_eye = self.model_eye(x,y[3])[:, :, :h, :w] - x_cond
                    MoE_input = torch.cat((MoE_input, output_eye), dim=1)
                else:
                    MoE_input = torch.cat((MoE_input,torch.zeros_like(x_cond)), dim=1)
                
                # print(MoE_input.shape) # torch.Size([16, 15, 1024, 1024])
                MoE_output = self.MoEmodel(MoE_input)
                content_loss, ssim_loss = self.estimation_loss(x_gt, MoE_output)
                loss = content_loss + ssim_loss
                # utils.logging.save_image(x_output, os.path.join(image_folder,f"{y[0]}_{id[0]}")) #保存名字id_标签

                if self.step % 10 == 0:
                    log_message = "step:{}, lr:{:.6f}, MSE loss:{:.4f}, SSIM loss:{:.4f}, total loss:{:.4f}".format(
                            self.step, self.scheduler.get_last_lr()[0], content_loss.item(), ssim_loss.item(), loss.item())

                    # 使用函数将消息写入文件
                    utils.logging.log(log_message,file_path=self.config.log_path)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.ema_helper.update(self.MoEmodel)
                
                if self.step % self.config.training.validation_freq == 0 and self.step != 0:
                    self.MoEmodel.eval()
                    val_PSNR = self.sample_validation_patches(val_loader, self.step, self.config.training.num_val) 
                    if val_PSNR > best_val_PSNR: # 如果提升是最高的，保存模型
                        best_val_PSNR = val_PSNR
                        utils.logging.log(f"==> Saving best model, PSNR lift: {val_PSNR}",file_path=self.config.log_path)
                        utils.logging.save_checkpoint({'step': self.step, 'epoch': epoch + 1,
                                                   'state_dict': self.MoEmodel.state_dict(),
                                                   'optimizer': self.optimizer.state_dict(),
                                                   'scheduler': self.scheduler.state_dict(),
                                                   'ema_helper': self.ema_helper.state_dict(),
                                                   'params': self.args,
                                                   'config': self.config,
                                                   'PSNR': best_val_PSNR},
                                                  filename=os.path.join(self.config.data.ckpt_dir, 'model_best_'+self.config.name+'_step'+str(self.step)+'_PSNR'+str(round(val_PSNR,2))))
                
            utils.logging.log(f"==> Saving checkpoint for epoch {epoch}",file_path=self.config.log_path)
            utils.logging.save_checkpoint({'step': self.step, 'epoch': epoch + 1,
                                                   'state_dict': self.MoEmodel.state_dict(),
                                                   'optimizer': self.optimizer.state_dict(),
                                                   'scheduler': self.scheduler.state_dict(),
                                                   'ema_helper': self.ema_helper.state_dict(),
                                                   'params': self.args,
                                                   'config': self.config},
                                                  filename=os.path.join(self.config.data.ckpt_dir, self.config.name+str(epoch)+'model_'+str(round(1-ssim_loss.item(),2))))
            self.scheduler.step()
    def estimation_loss(self,gt_img, pred_x):
        content_loss = self.l2_loss(pred_x, gt_img)
        ssim_loss = 1 - ssim(pred_x, gt_img, data_range=1.0).to(self.device)

        return content_loss, ssim_loss
    
    def sample_validation_patches(self, val_loader, step, num_samples): # 需要改一下，太费时间，也不清楚效果怎么样，需要有效输出PSNR比较
        image_folder = os.path.join(self.args.image_folder, self.config.name)
        self.MoEmodel.eval()
        
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
                # b, _, img_h, img_w = x.shape
                # img_h_32 = int(32 * np.ceil(img_h / 32.0)) # 调整为 32 的倍数
                # img_w_32 = int(32 * np.ceil(img_w / 32.0))
                # x = F.pad(x, (0, img_w_32 - img_w, 0, img_h_32 - img_h), 'reflect')

                # # print(x.shape,y.shape) #torch.Size([1, 6, 1024, 1024])
                # out = self.model(x.to(self.device),y)
                # pred_x = out["pred_x"]
                # pred_x = pred_x[:, :, :img_h, :img_w].to('cpu')
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                x = x.to(self.device)
                self.step += 1
                # 读取输入图像，并且确保是32的整数倍
                gt_x = x[:,3:,:,:]
                x_cond = x[:, :3, :, :]
                b, c, h, w = x_cond.shape
                img_h_32 = int(32 * np.ceil(h / 32.0))
                img_w_32 = int(32 * np.ceil(w / 32.0))
                x_cond = F.pad(x_cond, (0, img_w_32 - w, 0, img_h_32 - h), 'reflect')
                
                MoE_input = x_cond
                if '0w' in name:
                    output_white = self.model_white(x,y[0])[:, :, :h, :w] - x_cond
                    MoE_input = torch.cat((MoE_input, output_white), dim=1)
                else:
                    MoE_input = torch.cat((MoE_input,torch.zeros_like(x_cond)), dim=1)
                if '0s' in name:
                    output_smooth = self.model_smooth(x,y[1])[:, :, :h, :w] - x_cond
                    MoE_input = torch.cat((MoE_input, output_smooth), dim=1)
                else:
                    MoE_input = torch.cat((MoE_input,torch.zeros_like(x_cond)), dim=1)
                if '0f' in name:
                    output_face = self.model_face(x,y[2])[:, :, :h, :w] - x_cond
                    MoE_input = torch.cat((MoE_input, output_face), dim=1)
                else:
                    MoE_input = torch.cat((MoE_input,torch.zeros_like(x_cond)), dim=1)
                if '0e' in name:
                    output_eye = self.model_eye(x,y[3])[:, :, :h, :w] - x_cond
                    MoE_input = torch.cat((MoE_input, output_eye), dim=1)
                else:
                    MoE_input = torch.cat((MoE_input,torch.zeros_like(x_cond)), dim=1)
                
                # print(MoE_input.shape) # torch.Size([16, 15, 1024, 1024])
                pred_x = self.MoEmodel(MoE_input).to('cpu')
                
                # 增加测试PSNR的大小
                # gt_x = x[:, 3:, :, :]
                # input_x = x[:, :3, :, :]
                # PSNR_pred = utils.calculate_psnr(pred_x,gt_x)
                # PSNR_pred_list.append(PSNR_pred)
                # PSNR_ori = utils.calculate_psnr(input_x,gt_x)
                # # print(PSNR_ori)
                # PSNR_list.append(PSNR_pred-PSNR_ori)
                # print(gt_x.shape,input_x.shape,pred_x.shape)
                for ind in range(len(pred_x)):
                    PSNR_pred = PSNR(pred_x[ind].cpu().detach().numpy().transpose(1, 2, 0),gt_x[ind].cpu().detach().numpy().transpose(1, 2, 0))
                    PSNR_pred_list.append(PSNR_pred)
                    PSNR_ori = PSNR(x_cond[ind].cpu().detach().numpy().transpose(1, 2, 0),gt_x[ind].cpu().detach().numpy().transpose(1, 2, 0))
                    # print(PSNR_ori)
                    PSNR_list.append(PSNR_pred-PSNR_ori)
                positive_percentage = (sum(1 for x in PSNR_list if x > 0) / len(PSNR_list)) * 100
                
                utils.logging.save_image(pred_x, os.path.join(image_folder, str(step),f"y{y[0]}_{name[0]}"))

            utils.logging.log(f"validation num:{len(PSNR_list)},average PSNR: {np.mean(PSNR_pred_list)},average PSNR lift: {np.mean(PSNR_list)}, positive percentage:{positive_percentage}%",file_path=self.config.log_path)
        return np.mean(PSNR_list)
    ######### 测试用到以下
    def load_MoE_ckpt(self,resume):
        checkpoint = utils.logging.load_checkpoint(resume, None)
        self.MoEmodel.load_state_dict(checkpoint['state_dict'], strict=True)
        utils.logging.log("=> loaded checkpoint {}".format(resume),file_path=self.config.log_path)
    
    def restore(self,test_loader):
        # 读入并冻结模型
        print("=> loading Experts")
        self.load_expert_ckpt()
        
        print("=> loading MoE model")
        if os.path.isfile(self.args.resume):
            self.load_MoE_ckpt(self.args.resume)
            self.MoEmodel.eval()
        else:
            print('Pre-trained diffusion model path is missing!')

        image_folder = os.path.join(self.args.image_folder, self.args.test_name)
        
        total_samples = len(test_loader.dataset)
        print(f"=> Test dataset num :{total_samples}")

        with torch.no_grad():
            utils.logging.log(f"Processing Test!",file_path=self.config.log_path)
            for i, (x, y, name) in enumerate(test_loader):
                if i >= self.args.num_test:  # only apply our model to opt.num_test images.
                    print("already evaluate num:",self.args.num_test)
                    break
                start_time = time.time()
                # x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                x = x.to(self.device)
                # self.step += 1
                # 读取输入图像，并且确保是32的整数倍
                gt_x = x[:,3:,:,:]
                x_cond = x[:, :3, :, :]
                b, c, h, w = x_cond.shape
                img_h_32 = int(32 * np.ceil(h / 32.0))
                img_w_32 = int(32 * np.ceil(w / 32.0))
                x_cond = F.pad(x_cond, (0, img_w_32 - w, 0, img_h_32 - h), 'reflect')
                
                MoE_input = x_cond
                if '0w' in name:
                    output_white = self.model_white(x,y[0])[:, :, :h, :w] - x_cond
                    MoE_input = torch.cat((MoE_input, output_white), dim=1)
                else:
                    MoE_input = torch.cat((MoE_input,torch.zeros_like(x_cond)), dim=1)
                if '0s' in name:
                    output_smooth = self.model_smooth(x,y[1])[:, :, :h, :w] - x_cond
                    MoE_input = torch.cat((MoE_input, output_smooth), dim=1)
                else:
                    MoE_input = torch.cat((MoE_input,torch.zeros_like(x_cond)), dim=1)
                if '0f' in name:
                    output_face = self.model_face(x,y[2])[:, :, :h, :w] - x_cond
                    MoE_input = torch.cat((MoE_input, output_face), dim=1)
                else:
                    MoE_input = torch.cat((MoE_input,torch.zeros_like(x_cond)), dim=1)
                if '0e' in name:
                    output_eye = self.model_eye(x,y[3])[:, :, :h, :w] - x_cond
                    MoE_input = torch.cat((MoE_input, output_eye), dim=1)
                else:
                    MoE_input = torch.cat((MoE_input,torch.zeros_like(x_cond)), dim=1)
                
                # print(MoE_input.shape) # torch.Size([1, 15, 1024, 1024])
                pred_x = self.MoEmodel(MoE_input).to('cpu')
                
                utils.logging.save_image(pred_x, os.path.join(image_folder,f"{name[0]}"))
                print(f"processing image {name[0]}")
                print(time.time()-start_time)