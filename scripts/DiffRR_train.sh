############################## RetouchingOne ##################################
######### eye ##########
##@ CUDA_VISIBLE_DEVICES=0 /data/bin/python train.py --config RetouchingOne_eye.yml --resume ckpt/eye/model_latesteye_all.pth.tar
##@ CUDA_VISIBLE_DEVICES=1 /data/bin/python train.py --config RetouchingOne_eye_uncond.yml --resume ckpt/eye/model_latesteye_alluncond.pth.tar

######### face ##########
##@ CUDA_VISIBLE_DEVICES=2 /data/bin/python train.py --config RetouchingOne_face.yml --resume ckpt/face/model_latestface_all.pth.tar
##@ CUDA_VISIBLE_DEVICES=3 /data/bin/python train.py --config RetouchingOne_face_uncond.yml --resume ckpt/face/model_latestface_alluncond.pth.tar

######### white ##########
##@ CUDA_VISIBLE_DEVICES=0 /data/bin/python train.py --config RetouchingOne_white.yml --resume ckpt/white/model_latestwhite_all.pth.tar
##@ CUDA_VISIBLE_DEVICES=1 /data/bin/python train.py --config RetouchingOne_white_uncond.yml --resume ckpt/white/model_latestwhite_alluncond.pth.tar

######### smooth ##########
##@ CUDA_VISIBLE_DEVICES=2 python train.py --config RetouchingOne_smooth.yml
##@ CUDA_VISIBLE_DEVICES=2 python train.py --config RetouchingOne_smooth_uncond.yml --resume ckpt/smooth/model_latestsmooth_alluncond.pth.tar

#################################### LF ######################
## CUDA_VISIBLE_DEVICES=1 python train.py --config RetouchingOne_smooth_DERestormer.yml # 其实是Simple


# 上面的程度全都是反的！！！不过不影响性能
##################################### LF ##############################
CUDA_VISIBLE_DEVICES=2 python train.py --config RetouchingOne_white_DESimple.yml 

################################### LFuncond ###################
## CUDA_VISIBLE_DEVICES=0 python train.py --config RetouchingOne_smoothuncond_DESimple.yml # 万一程度训不出来,调整了程度




#################################### uncond + HF 暂训50个 ####################################
##@ python train.py --config uncond_HF/RetouchingOne_whiteuncond_HF.yml #26四张
##@ python train.py --config uncond_HF/RetouchingOne_smoothuncond_HF.yml #50epoch三天
##@ CUDA_VISIBLE_DEVICES=0,1 python train.py --config uncond_HF/RetouchingOne_eyeuncond_HF.yml 
##@ CUDA_VISIBLE_DEVICES=2,3 python train.py --config uncond_HF/RetouchingOne_faceuncond_HF.yml 

#################################### info1（很差的回归模型预测的结果作为条件）暂训50个 ##################################################
## CUDA_VISIBLE_DEVICES=1 python train.py --config info1_config/RetouchingOne_white_DEinfo1.yml 
## CUDA_VISIBLE_DEVICES=2 python train.py --config info1_config/RetouchingOne_eye_DEinfo1.yml 
## CUDA_VISIBLE_DEVICES=3 python train.py --config info1_config/RetouchingOne_smooth_DEinfo1.yml -- resume /newdata/DiffRR/ckpt/smooth/model_latestsmooth_allDEinfo1.pth.tar #12.3
## CUDA_VISIBLE_DEVICES=0 python train.py --config info1_config/RetouchingOne_face_DEinfo1.yml 

#################################### info1（ResNet预测作为条件）训100个 ##################################################
##@ CUDA_VISIBLE_DEVICES=0 python train.py --config info1_ResNet/RetouchingOne_eye_DEinfo1_Res.yml # 达到63了断掉
##@ CUDA_VISIBLE_DEVICES=1 python train.py --config info1_ResNet/RetouchingOne_smooth_DEinfo1_Res.yml # 达到50了断掉
##@ CUDA_VISIBLE_DEVICES=0 python train.py --config info1_ResNet/RetouchingOne_face_DEinfo1_Res.yml # 达到50了断掉
##@ CUDA_VISIBLE_DEVICES=2 python train.py --config info1_ResNet/RetouchingOne_white_DEinfo1_Res.yml # 达到50了断掉

##################################### info2 iAFF （ResNet预测作为条件） degree ########################################
## CUDA_VISIBLE_DEVICES=3 python train.py --config DE_info2/RetouchingOne_eye_info2_d_iAFF.yml  
## CUDA_VISIBLE_DEVICES=2 python train.py --config DE_info2/RetouchingOne_face_info2_d_iAFF.yml 
## CUDA_VISIBLE_DEVICES=0 python train.py --config DE_info2/RetouchingOne_white_info2_d_iAFF.yml 
## CUDA_VISIBLE_DEVICES=1 python train.py --config DE_info2/RetouchingOne_smooth_info2_d_iAFF.yml #似乎是旧的跑完了,看起来还可以

##################################### info3 iAFF+GradCAM ！！！ (ResNet5作为degree和CAM来源) ################################
##@ CUDA_VISIBLE_DEVICES=3 python train.py --config DE_infoCAM/RetouchingOne_white_info2_CAM_iAFF.yml 
##@ CUDA_VISIBLE_DEVICES=1 python train.py --config DE_infoCAM/RetouchingOne_eye_info2_CAM_iAFF.yml 
##@ CUDA_VISIBLE_DEVICES=1 python train.py --config DE_infoCAM/RetouchingOne_smooth_info2_CAM_iAFF.yml 
##@ CUDA_VISIBLE_DEVICES=1 python train.py --config DE_infoCAM/RetouchingOne_face_info2_CAM_iAFF.yml 

###################################### DiffLL(baseline) ##################################################
## CUDA_VISIBLE_DEVICES=5 python train.py --config DiffLL/RetouchingOne_eye_DiffLL.yml --resume /newdata/DiffRR/ckpt/eye/DiffLL/model_latesteye_DiffLL.pth.tar
## CUDA_VISIBLE_DEVICES=1 python train.py --config DiffLL/RetouchingOne_face_DiffLL.yml --resume /newdata/DiffRR/ckpt/face/DiffLL/model_latestface_DiffLL.pth.tar # 15.1
## CUDA_VISIBLE_DEVICES=2 python train.py --config DiffLL/RetouchingOne_white_DiffLL.yml --resume /newdata/DiffRR/ckpt/white/DiffLL/model_latestwhite_DiffLL.pth.tar
## CUDA_VISIBLE_DEVICES=0 python train.py --config DiffLL/RetouchingOne_smooth_DiffLL.yml 

###################################### difrr: info3+HF 100求求了千万搞好点）！！！ 2张卡 ##################################
# CUDA_VISIBLE_DEVICES=0,1 python train.py --config diffrr/RetouchingOne_face_diffrr.yml # 27.01
# CUDA_VISIBLE_DEVICES=2,3 python train.py --config diffrr/RetouchingOne_white_diffrr.yml --resume /newdata/DiffRR/ckpt/white/diffrr/model_latestwhite_diffrr.pth.tar # 21.23
# CUDA_VISIBLE_DEVICES=2,3 python train.py --config diffrr/RetouchingOne_smooth_diffrr.yml # 27.23
# CUDA_VISIBLE_DEVICES=0,1 python train.py --config diffrr/RetouchingOne_eye_diffrr.yml #21.01



######################################### DiffLL处理multi数据  15环境要求(venv)(base)
# CUDA_VISIBLE_DEVICES=5 python train.py --config Multi/RetouchingOne_multi_DiffLL.yml --resume /newdata/DiffRR/ckpt/multi/DiffLL/model_latestmulti_DiffLL.pth.tar #15.5



######################################### info2+HF(可能是最好的)！！！ ##################################
# CUDA_VISIBLE_DEVICES=0,1 python train.py --config DiffRR_info2/RetouchingOne_eye_diffrr_info2.yml # 21
# CUDA_VISIBLE_DEVICES=0,1 python train.py --config DiffRR_info2/RetouchingOne_face_diffrr_info2.yml --resume /newdata/DiffRR/ckpt/face/diffrr_info2/model_latestface_diffrr_info2.pth.tar #26
# CUDA_VISIBLE_DEVICES=2,3 python train.py --config DiffRR_info2/RetouchingOne_white_diffrr_info2.yml # 21
# CUDA_VISIBLE_DEVICES=1,0 python train.py --config DiffRR_info2/RetouchingOne_smooth_diffrr_info2.yml # 27

######################################### 消融实验！！！ ##################################
############### 无HF的已经有了，是info2 ####################
############### 无IDEM的 ####################
#@ CUDA_VISIBLE_DEVICES=7 python train.py --config /newdata/DiffRR/configs/ablation_IDEM/RetouchingOne_eye_ablation_IDEM.yml #15(venv)(base)
#@ CUDA_VISIBLE_DEVICES=5 python train.py --config /newdata/DiffRR/configs/ablation_IDEM/RetouchingOne_face_ablation_IDEM.yml #15
#@ CUDA_VISIBLE_DEVICES=2 python train.py --config /newdata/DiffRR/configs/ablation_IDEM/RetouchingOne_white_ablation_IDEM.yml #26
#@ CUDA_VISIBLE_DEVICES=3 python train.py --config /newdata/DiffRR/configs/ablation_IDEM/RetouchingOne_smooth_ablation_IDEM.yml #27
############### 无degree的 ####################
CUDA_VISIBLE_DEVICES=1 python train.py --config /newdata/DiffRR/configs/ablation_degree/RetouchingOne_eye_ablation_degree.yml --resume /newdata/DiffRR/ckpt/eye/ablation_d/model_latesteye_ablation_d.pth.tar #27死了
# CUDA_VISIBLE_DEVICES=0 python train.py --config /newdata/DiffRR/configs/ablation_degree/RetouchingOne_face_ablation_degree.yml #12
# CUDA_VISIBLE_DEVICES=2 python train.py --config /newdata/DiffRR/configs/ablation_degree/RetouchingOne_white_ablation_degree.yml #12
# CUDA_VISIBLE_DEVICES=3 python train.py --config /newdata/DiffRR/configs/ablation_degree/RetouchingOne_smooth_ablation_degree.yml #12