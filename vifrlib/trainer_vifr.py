## This file has been taken from DECA and modified ##

import os, sys
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from time import time
from skimage.io import imread
import cv2
import pickle
from loguru import logger
from datetime import datetime
from tqdm import tqdm
import time
from .utils.renderer import SRenderY
from .models.encoders import ResnetEncoder
# from .models.FLAME import FLAME, FLAMETex
from .models.decoders import Generator
from .utils import util
from .utils.rotation_converter import batch_euler2axis
from .datasets import datasets
from .utils.config import cfg
torch.backends.cudnn.benchmark = True
from .utils import lossfunc, lossfunc_new
from .datasets import build_datasets
# from perlin_noise import rand_perlin_2d_octaves
# from stylegan2.model import Generator
# from img_2_tex import mesh_angle, tex_correction
from prefetch_generator import BackgroundGenerator

# new
# from .models.hifi3dpp import ParametricFaceModel



class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class Trainer(object):
    def __init__(self, model, config=None, device='cuda:0'):
        if config is None:
            self.cfg = cfg
        else:
            self.cfg = config
        self.device = device
        self.batch_size = self.cfg.dataset.batch_size
        self.image_size = self.cfg.dataset.image_size
        # self.uv_size = self.cfg.model.uv_size
        # self.K = self.cfg.dataset.K
        self.fm_model_file = self.cfg.model.fm_model_file
        self.unwrap_info_file = self.cfg.model.unwrap_info_file
        self.texgan_model_file = self.cfg.model.texgan_model_file
        self.net_recon_path = self.cfg.model.net_recon_path
        self.E_hifi3d_backbone = self.cfg.model.E_hifi3d_backbone
        self.net_recog_path = self.cfg.model.net_recog_path
        self.net_recog = self.cfg.model.net_recog 
        self.param_list = self.cfg.model.param_list
        self.lmk_num = self.cfg.loss.lmk_num
        
        
        
        
        # # self.fm_model_file = self.cfg.model
        # #     'unwrap_info_file': os.path.join(topo_dir, 'unwrap_1024_info.mat'),
        # #     'camera_distance': 10.,
        # #     'focal': 1015.,
        # #     'center': 112.,
        # #     'znear': 5.,
        # #     'zfar': 15.,
        # #     # texture gan
        # #     'texgan_model_file': os.path.join(cpk_dir, f'texgan_model/{texgan_model_name}'),
        # #     # deep3d nn inference model
        # #     'net_recon': 'resnet50',
        # #     'net_recon_path': os.path.join(cpk_dir, 'deep3d_model/epoch_latest.pth'),
        # #     # recognition model
        # #     'net_recog': 'r50',
        # #     'net_recog_path': os.path.join(cpk_dir, 'arcface_model/ms1mv3_arcface_r50_fp16_backbone.pth'),
        # #     # vgg model
        # #     'net_vgg_path': os.path.join(cpk_dir, 'vgg_model/vgg16.pt'),
        # # }
        # # self.args_s2_search_uvtex_spherical_fixshape = {
        # #     'w_feat': 10.0,
        # #     'w_color': 10.0,
        # #     'w_vgg': 100.0,
        # #     'w_reg_latent': 0.05,
        # #     'initial_lr': 0.1,
        # #     'lr_rampdown_length': 0.25,
        # #     'total_step': 100,
        # #     'print_freq': 5,
        # #     'visual_freq': 10,
        # # }
        # # self.args_s3_optimize_uvtex_shape_joint = {
        # #     'w_feat': 0.2,
        # #     'w_color': 1.6,
        # #     'w_reg_id': 2e-4,
        # #     'w_reg_exp': 1.6e-3,
        # #     'w_reg_gamma': 10.0,
        # #     'w_reg_latent': 0.05,
        # #     'w_lm': 2e-3,
        # #     'initial_lr': 0.01,
        # #     'tex_lr_scale': 1.0 if loose_tex else 0.05,
        # #     'lr_rampdown_length': 0.4,
        # #     'total_step': 200,
        # #     'print_freq': 10,
        # #     'visual_freq': 20,
        # # }

        # # self.args_names = ['model', 's2_search_uvtex_spherical_fixshape', 's3_optimize_uvtex_shape_joint']

        # # parametric face model
        # self.facemodel = ParametricFaceModel(fm_model_file=self.fm_model_file,
        #                                      unwrap_info_file=self.unwrap_info_file,
        #                                      camera_distance=self.camera_distance,
        #                                      focal=self.focal,
        #                                      center=self.center,
        #                                      lm86=True if self.lmk_num==86 else False,
        #                                      device=self.device)

        # # texture gan
        # self.tex_gan = texgan.TextureGAN(model_path=self.args_model['texgan_model_file'], device=device)

        # # deep3d nn reconstruction model
        # fc_info = {
        #     'id_dims': self.facemodel.id_dims,
        #     'exp_dims': self.facemodel.exp_dims,
        #     'tex_dims': self.facemodel.tex_dims
        # }
        # self.net_recon_deep3d = define_net_recon_deep3d(net_recon=self.args_model['net_recon'],
        #                                                 use_last_fc=False,
        #                                                 fc_dim_dict=fc_info,
        #                                                 pretrained_path=self.args_model['net_recon_path'])
        # self.net_recon_deep3d = self.net_recon_deep3d.eval().requires_grad_(False)

        # # renderer
        # fov = 2 * np.arctan(self.args_model['center'] / self.args_model['focal']) * 180 / np.pi
        # self.renderer = MeshRenderer(fov=fov,
        #                              znear=self.args_model['znear'],
        #                              zfar=self.args_model['zfar'],
        #                              rasterize_size=int(2 * self.args_model['center']))

        # # the recognition model
        # self.net_recog = define_net_recog(net_recog=self.args_model['net_recog'],
        #                                   pretrained_path=self.args_model['net_recog_path'])
        # self.net_recog = self.net_recog.eval().requires_grad_(False)

        # # the vgg model
        # with dnnlib.util.open_url(self.args_model['net_vgg_path']) as f:
        #     self.net_vgg = torch.jit.load(f).eval()

        # # coeffs and latents
        # self.pred_coeffs = None
        # self.pred_latents_w = None
        # self.pred_latents_z = None

        # self.to(device)
        # self.device = device














        # # 原
        # if config is None:
        #     self.cfg = cfg
        # else:
        #     self.cfg = config
        # self.device = device
        # self.batch_size = self.cfg.dataset.batch_size
        # self.image_size = self.cfg.dataset.image_size
        # self.uv_size = self.cfg.model.uv_size
        # self.K = self.cfg.dataset.K
        # # training stage: coarse and detail
        # self.train_detail = self.cfg.train.train_detail
        # self.train_albedo = self.cfg.train.train_albedo
        # self.train_arcface_other_params = self.cfg.train_other_params
        # self.id_loss_2 = lossfunc.VGGFace2Loss(pretrained_model=os.path.join('data/resnet50_ft_weight.pkl'))
        
        # if self.train_albedo:

        #     self.albedoGAN = Generator(1024, 512, 8, channel_multiplier=2)
        #     self.albedoGAN = torch.nn.DataParallel(self.albedoGAN)
        #     self.albedoGAN.to(device)
        #     ckpt_path = 'stylegan2/checkpoint/albedogan.pt'
        #     ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        #     self.albedoGAN.load_state_dict(ckpt["g"], strict=True)

        #     self.stylegan2_gen = Generator(1024, 512, 8, channel_multiplier=2)
        #     self.stylegan2_gen.to(device)
        #     ckpt_path = 'stylegan2/training_logs/checkpoint/stylegan2_ffhq.pt'
        #     ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        #     self.stylegan2_gen.load_state_dict(ckpt["g_ema"], strict=True)
        #     self.mean_latent = self.stylegan2_gen.mean_latent(4096)

        #     self.iscrop = True
        #     self.detector = 'fan'
        #     self.sample_step = 10

        #     self.sample_images = datasets.TestData_stylegan(self.stylegan2_gen, batch_size=1, 
        #                                  iscrop=self.iscrop, face_detector=self.detector, 
        #                                  sample_step=self.sample_step, crop_size=1024)
            
        #     # useTex = True
        #     # extractTex = True
        #     # rasterizer_type = 'pytorch3d'
        #     # device = 'cuda'

        #     # deca_cfg.model.use_tex = useTex
        #     # deca_cfg.rasterizer_type = rasterizer_type
        #     # deca_cfg.model.extract_tex = extractTex
        #     # self.deca = DECA(config = deca_cfg, device=device)
            


        # vifr model
        self.vifr = model.to(self.device)
        # 设置优化器
        self.opt = torch.optim.Adam(self.vifr.E_hifi3d.parameters(),lr=self.cfg.train.lr,amsgrad=False)
        # self.configure_optimizers()
        self.load_checkpoint()

        # initialize loss  
        # # initialize loss   
        # if self.train_detail:     
        #     self.mrf_loss = lossfunc.IDMRFLoss()
        #     self.face_attr_mask = util.load_local_mask(image_size=self.cfg.model.uv_size, mode='bbx')
        # else:
        #     self.id_loss = lossfunc.VGGFace2Loss(pretrained_model=self.cfg.model.fr_model_path)      
        
        logger.add(os.path.join(self.cfg.output_dir, self.cfg.train.log_dir, 'train.log'))
        if self.cfg.train.write_summary:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=os.path.join(self.cfg.output_dir, self.cfg.train.log_dir))
    
    # def configure_optimizers(self):

    #     if self.train_arcface_other_params:
    #         self.opt = torch.optim.Adam(
    #             self.deca.MICA_otherParamModel.parameters(),
    #             lr=self.cfg.train.lr,
    #             amsgrad=False)
            
    #     if self.train_detail and not self.train_albedo:
    #         self.opt = torch.optim.Adam(
    #                             list(self.deca.E_detail.parameters()) + \
    #                             list(self.deca.D_detail.parameters())  ,
    #                             lr=self.cfg.train.lr,
    #                             amsgrad=False)
            
    #     elif not self.train_detail:
    #         self.opt = torch.optim.Adam(
    #                                 self.deca.E_hifi3d.parameters(),
    #                                 lr=self.cfg.train.lr,
    #                                 amsgrad=False)
        
    #     elif self.train_albedo:
    #         self.opt = torch.optim.Adam(
    #                                 self.albedoGAN.parameters(),
    #                                 lr=self.cfg.train.lr,
    #                                 amsgrad=False)

        

    def load_checkpoint(self):
        model_dict = self.vifr.model_dict()
        # resume training, including model weight, opt, steps
        # import ipdb; ipdb.set_trace()
        if self.cfg.train.resume and os.path.exists(os.path.join(self.cfg.output_dir, 'model.tar')):
            checkpoint = torch.load(os.path.join(self.cfg.output_dir, 'model.tar'))
            for key in model_dict.keys():
                if key in checkpoint.keys():
                    util.copy_state_dict(model_dict[key], checkpoint[key])
            util.copy_state_dict(self.opt.state_dict(), checkpoint['opt'])
            self.global_step = checkpoint['global_step']
            logger.info(f"resume training from {os.path.join(self.cfg.output_dir, 'model.tar')}")
            logger.info(f"training start from step {self.global_step}")
        # load model weights only
        elif os.path.exists(self.cfg.pretrained_modelpath):
            checkpoint = torch.load(self.cfg.pretrained_modelpath)
            key = 'E_hifi3d'
            util.copy_state_dict(model_dict[key], checkpoint[key])
            self.global_step = 0
        else:
            logger.info('model path not found, start training from scratch')
            self.global_step = 0

    def training_step(self, batch, step):
        self.vifr.train()
        # if self.train_detail:
        #     self.deca.E_flame.eval()
        # [B, K, 3, size, size] ==> [BxK, 3, size, size]
        image = batch['image'].to(self.device); image = image.view(-1, image.shape[-3], image.shape[-2], image.shape[-1]) 
        gt_lmk = batch['landmark'].to(self.device); gt_lmk = gt_lmk.view(-1, gt_lmk.shape[-2], gt_lmk.shape[-1])
        gt_mask = batch['mask'].to(self.device); gt_mask = gt_mask.view(-1, image.shape[-2], image.shape[-1]) 
        gt_arcface = batch['arcface'].to(self.device); gt_arcface = gt_arcface.view(-1, gt_arcface.shape[-3], gt_arcface.shape[-2], gt_arcface.shape[-1]) 

        #-- encoder
        codedict = self.vifr.encode(torchvision.transforms.Resize(224)(image))
        #-- decoder
        rendering = True if self.cfg.loss.photo>0 else False
        opdict = self.vifr.decode(codedict, rendering = rendering, vis_lmk=False, return_vis=False, use_detail=False)
        # -------------------------------------------对opdict补充
        opdict['image'] = image
        opdict['gt_lmk'] = gt_lmk
        # 获取渲染后的图像图像
        if self.cfg.loss.photo > 0.:
            # pred_face_img = opdict['render_face']  * opdict['render_face_mask'] + (1 - opdict['render_face_mask']) * opdict['image']
            # opdict[]
            pass
            # #------ rendering
            # # mask
            # mask_face_eye = F.grid_sample(self.deca.uv_face_eye_mask.expand(batch_size,-1,-1,-1), opdict['grid'].detach(), align_corners=False) 
            # # images
            # predicted_images = opdict['rendered_images']*mask_face_eye*opdict['alpha_images']
            # opdict['predicted_images'] = predicted_images
        
        # ------------------------------------------计算各Loss
        losses = {}
        # 关键点损失
        losses['landmark'] = 0
        ## 嘴唇关键点距离损失
        losses['lip_distance'] = 0
        ## 眼皮关键点距离损失
        losses['eye_distance'] = 0
        # 光度损失
        losses['photometric_texture'] = 0
        # 身份损失
        losses['identity'] = 0
        # 正则化损失
        losses['id_reg'] = (torch.sum(codedict['id']**2)/2)*self.cfg.loss.reg_id
        losses['exp_reg'] = (torch.sum(codedict['exp']**2)/2)*self.cfg.loss.reg_exp
        losses['tex_reg'] = (torch.sum(codedict['tex']**2)/2)*self.cfg.loss.reg_tex
        losses['gamma_reg'] = ((torch.mean(codedict['gamma'], dim=2)[:,:,None] - codedict['gamma'])**2).mean()*self.cfg.loss.reg_gamma
        # 总损失
        all_loss = 0.
        losses_key = losses.keys()
        for key in losses_key:
            all_loss = all_loss + losses[key]
        losses['all_loss'] = all_loss
        # print("计算完损失函数值")
        return losses, opdict
        
        
        if self.cfg.loss.shape_consistency or self.cfg.loss.detail_consistency:
            '''
            make sure s0, s1 is something to make shape close
            the difference from ||so - s1|| is 
            the later encourage s0, s1 is cloase in l2 space, but not really ensure shape will be close
            待开发
            '''
            pass
            new_order = np.array([np.random.permutation(self.K) + i * self.K for i in range(self.batch_size)])
            new_order = new_order.flatten()
            shapecode = codedict['shape']
            # mica_shapecode = codedict['mica_shape']
            expcode = codedict['exp']
            # posecode = codedict['pose']
            if self.train_detail:
                detailcode = codedict['detail']
                detailcode_new = detailcode[new_order]
                codedict['detail'] = torch.cat([detailcode, detailcode_new], dim=0)
                codedict['shape'] = torch.cat([shapecode, shapecode], dim=0)
                codedict['exp'] = torch.cat([expcode, expcode], dim=0)
                # codedict['pose'] = torch.cat([posecode, posecode], dim=0)
            else:
                shapecode_new = shapecode[new_order]
                expcode_new = expcode[new_order]
                # posecode_new = posecode[new_order]
                codedict['shape'] = torch.cat([shapecode, shapecode_new], dim=0)
                codedict['exp'] = torch.cat([expcode, expcode_new], dim=0)
                # codedict['pose'] = torch.cat([posecode, posecode_new], dim=0)
            for key in ['tex', 'cam', 'pose', 'light', 'images','mica_shape']:
                if key in codedict:
                    code = codedict[key]
                    codedict[key] = torch.cat([code, code], dim=0)
            ## append gt
            images = torch.cat([images, images], dim=0)# images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1]) 
            lmk = torch.cat([lmk, lmk], dim=0) #lmk = lmk.view(-1, lmk.shape[-2], lmk.shape[-1])
            masks = torch.cat([masks, masks], dim=0)
            pass

        batch_size = images.shape[0]

        ###training coarse model
        if not self.train_detail:
            pass
            #-- decoder
            rendering = True if self.cfg.loss.photo>0 else False
            opdict = self.deca.decode(codedict, rendering = rendering, vis_lmk=False, return_vis=False, use_detail=False)
            opdict['images'] = images
            opdict['lmk'] = lmk

            if self.cfg.loss.photo > 0.:
                #------ rendering
                # mask
                mask_face_eye = F.grid_sample(self.deca.uv_face_eye_mask.expand(batch_size,-1,-1,-1), opdict['grid'].detach(), align_corners=False) 
                # images
                predicted_images = opdict['rendered_images']*mask_face_eye*opdict['alpha_images']
                opdict['predicted_images'] = predicted_images

            #### ----------------------- Losses
            losses = {}
            
            ############################# base shape
            # predicted_landmarks = opdict['landmarks2d']
            # if self.cfg.loss.useWlmk:
            #     losses['landmark'] = lossfunc_new.weighted_landmark_loss(predicted_landmarks, lmk)*self.cfg.loss.lmk
            # else:    
            #     losses['landmark'] = lossfunc_new.landmark_loss(predicted_landmarks, lmk)*self.cfg.loss.lmk
            # if self.cfg.loss.eyed > 0.:
            #     losses['eye_distance'] = lossfunc_new.eyed_loss(predicted_landmarks, lmk)*self.cfg.loss.eyed
            # if self.cfg.loss.lipd > 0.:
            #     losses['lip_distance'] = lossfunc_new.lipd_loss(predicted_landmarks, lmk)*self.cfg.loss.lipd
            # if self.cfg.loss.photo > 0.:
            #     if self.cfg.loss.useSeg:
            #         masks = masks[:,None,:,:]
            #     else:
            #         masks = mask_face_eye*opdict['alpha_images']
            #     losses['photometric_texture'] = (masks*(predicted_images - images).abs()).mean()*self.cfg.loss.photo
# 改造
            if self.cfg.loss.lmk_num == 68:
                predicted_landmarks = opdict['landmarks2d']
                if self.cfg.loss.useWlmk:
                    losses['landmark'] = lossfunc_new.weighted_landmark_68_loss(predicted_landmarks, lmk) * self.cfg.loss.lmk
                    #################新新新新新新新新新新新新新新新新新新新新新新新新新新新新新新新新新新新####################
                    losses['mouth_kc'] = lossfunc_new.mouth_kc_loss(predicted_landmarks[:, 48:68, :], lmk[:, 48:68, :], sigma=0.3) *self.cfg.loss.mouthkc
                    #################新新新新新新新新新新新新新新新新新新新新新新新新新新新新新新新新新新新####################
                else:
                    losses['landmark'] = lossfunc_new.landmark_loss(predicted_landmarks, lmk) * self.cfg.loss.lmk
                if self.cfg.loss.eyed > 0.:
                    losses['eye_distance'] = lossfunc_new.eyed_loss(predicted_landmarks, lmk) * self.cfg.loss.eyed
                if self.cfg.loss.lipd > 0.:
                    losses['lip_distance'] = lossfunc_new.lipd_loss(predicted_landmarks, lmk) * self.cfg.loss.lipd
            elif self.cfg.loss.lmk_num == 191:
                predicted_landmarks = opdict['landmarks2d_191']
                if self.cfg.loss.useWlmk:
                    losses['landmark'] = lossfunc_new.weighted_landmark_191_loss(predicted_landmarks, lmk) * self.cfg.loss.lmk
                else:
                    losses['landmark'] = lossfunc_new.landmark_loss(predicted_landmarks, lmk) * self.cfg.loss.lmk
                if self.cfg.loss.eyed > 0.:
                    losses['eye_distance'] = lossfunc_new.eyed_loss_191(predicted_landmarks, lmk) * self.cfg.loss.eyed
                if self.cfg.loss.lipd > 0.:
                    losses['lip_distance'] = lossfunc_new.lipd_loss_191(predicted_landmarks, lmk) * self.cfg.loss.lipd
            elif self.cfg.loss.lmk_num == 235:
                predicted_landmarks = opdict['landmarks2d_235']
                if self.cfg.loss.useWlmk:
                    losses['landmark'] = lossfunc_new.weighted_landmark_235_loss(predicted_landmarks, lmk) * self.cfg.loss.lmk
                else:
                    losses['landmark'] = lossfunc_new.landmark_loss(predicted_landmarks, lmk) * self.cfg.loss.lmk
                if self.cfg.loss.eyed > 0.:
                    losses['eye_distance'] = lossfunc_new.eyed_loss_235(predicted_landmarks, lmk) * self.cfg.loss.eyed
                if self.cfg.loss.lipd > 0.:
                    losses['lip_distance'] = lossfunc_new.lipd_loss_235(predicted_landmarks, lmk) * self.cfg.loss.lipd
            else:
                print("please check cfg.loss.lmk_num's value!")

            if self.cfg.loss.id > 0.:
                shading_images = self.deca.render.add_SHlight(opdict['normal_images'], codedict['light'].detach())
                albedo_images = F.grid_sample(opdict['albedo'].detach(), opdict['grid'], align_corners=False)
                overlay = albedo_images*shading_images*mask_face_eye + images*(1-mask_face_eye)
                losses['identity'] = self.id_loss(overlay, images) * self.cfg.loss.id
            
            losses['shape_reg'] = (torch.sum(codedict['shape']**2)/2)*self.cfg.loss.reg_shape
            losses['expression_reg'] = (torch.sum(codedict['exp']**2)/2)*self.cfg.loss.reg_exp
            losses['tex_reg'] = (torch.sum(codedict['tex']**2)/2)*self.cfg.loss.reg_tex
            losses['light_reg'] = ((torch.mean(codedict['light'], dim=2)[:,:,None] - codedict['light'])**2).mean()*self.cfg.loss.reg_light
            if self.cfg.model.jaw_type == 'euler':
                # import ipdb; ipdb.set_trace()
                # reg on jaw pose
                losses['reg_jawpose_roll'] = (torch.sum(codedict['euler_jaw_pose'][:,-1]**2)/2)*100.
                losses['reg_jawpose_close'] = (torch.sum(F.relu(-codedict['euler_jaw_pose'][:,0])**2)/2)*10.
        
        all_loss = 0.
        losses_key = losses.keys()
        for key in losses_key:
            all_loss = all_loss + losses[key]
        losses['all_loss'] = all_loss
        return losses, opdict

        
    def validation_step(self):
        self.deca.eval()
        try:
            batch = next(self.val_iter)
        except:
            self.val_iter = iter(self.val_dataloader)
            batch = next(self.val_iter)
        images = batch['image'].to(self.device); images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1]) 
        arcfaces = batch['arcface'].to(self.device); arcfaces = arcfaces.view(-1, arcfaces.shape[-3], arcfaces.shape[-2], arcfaces.shape[-1]) 
        with torch.no_grad():
            codedict = self.deca.encode(images,arcfaces)
            opdict, visdict = self.deca.decode(codedict)
        savepath = os.path.join(self.cfg.output_dir, self.cfg.train.val_vis_dir, f'{self.global_step:08}.jpg')
        grid_image = util.visualize_grid(visdict, savepath, return_gird=True)
        self.writer.add_image('val_images', (grid_image/255.).astype(np.float32).transpose(2,0,1), self.global_step)
        self.deca.train()

    def evaluate(self):
        ''' NOW validation 
        '''
        os.makedirs(os.path.join(self.cfg.output_dir, 'NOW_validation'), exist_ok=True)
        savefolder = os.path.join(self.cfg.output_dir, 'NOW_validation', f'step_{self.global_step:08}') 
        os.makedirs(savefolder, exist_ok=True)
        self.deca.eval()
        # run now validation images
        from .datasets.now import NoWDataset
        dataset = NoWDataset(scale=(self.cfg.dataset.scale_min + self.cfg.dataset.scale_max)/2)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False,
                            num_workers=8,
                            pin_memory=True,
                            drop_last=False)
        faces = self.deca.flame.faces_tensor.cpu().numpy()
        for i, batch in enumerate(tqdm(dataloader, desc='now evaluation ')):
            images = batch['image'].to(self.device)
            imagename = batch['imagename']
            with torch.no_grad():
                codedict = self.deca.encode(images)
                _, visdict = self.deca.decode(codedict)
                codedict['exp'][:] = 0.
                codedict['pose'][:] = 0.
                opdict, _ = self.deca.decode(codedict)
            #-- save results for evaluation
            verts = opdict['verts'].cpu().numpy()
            landmark_51 = opdict['landmarks3d_world'][:, 17:]
            landmark_7 = landmark_51[:,[19, 22, 25, 28, 16, 31, 37]]
            landmark_7 = landmark_7.cpu().numpy()
            for k in range(images.shape[0]):
                os.makedirs(os.path.join(savefolder, imagename[k]), exist_ok=True)
                # save mesh
                util.write_obj(os.path.join(savefolder, f'{imagename[k]}.obj'), vertices=verts[k], faces=faces)
                # save 7 landmarks for alignment
                np.save(os.path.join(savefolder, f'{imagename[k]}.npy'), landmark_7[k])
                for vis_name in visdict.keys(): #['inputs', 'landmarks2d', 'shape_images']:
                    if vis_name not in visdict.keys():
                        continue
                    # import ipdb; ipdb.set_trace()
                    image = util.tensor2image(visdict[vis_name][k])
                    name = imagename[k].split('/')[-1]
                    # print(os.path.join(savefolder, imagename[k], name + '_' + vis_name +'.jpg'))
                    cv2.imwrite(os.path.join(savefolder, imagename[k], name + '_' + vis_name +'.jpg'), image)
            # visualize results to check
            util.visualize_grid(visdict, os.path.join(savefolder, f'{i}.jpg'))

        ## then please run main.py in https://github.com/soubhiksanyal/now_evaluation, it will take around 30min to get the metric results
        self.deca.train()

    def prepare_data(self):
        # self.cfg.dataset['image_size'] = 1024
        self.cfg.dataset['image_size'] = 224
        self.train_dataset = build_datasets.build_train(self.cfg)
        self.cfg.dataset['image_size'] = 224
        self.val_dataset = build_datasets.build_val(self.cfg)
        logger.info('---- training data numbers: ', len(self.train_dataset))

        self.train_dataloader = DataLoaderX(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                            num_workers=self.cfg.dataset.num_workers,
                            pin_memory=True,
                            drop_last=True)
        self.train_iter = iter(self.train_dataloader)
        self.val_dataloader = DataLoaderX(self.val_dataset, batch_size=8, shuffle=True,
                            num_workers=8,
                            pin_memory=True,
                            drop_last=False)
        self.val_iter = iter(self.val_dataloader)

    def fit(self):
        self.prepare_data()

        iters_every_epoch = int(len(self.train_dataset)/self.batch_size)
        start_epoch = self.global_step//iters_every_epoch
        # self.evaluate()
        # input('wait')
        for epoch in range(start_epoch, self.cfg.train.max_epochs):
            # for step, batch in enumerate(tqdm(self.train_dataloader, desc=f"Epoch: {epoch}/{self.cfg.train.max_epochs}")):
            for step in tqdm(range(iters_every_epoch), desc=f"Epoch[{epoch+1}/{self.cfg.train.max_epochs}]"):
                if epoch*iters_every_epoch + step < self.global_step:
                    continue
                try:
                    batch = next(self.train_iter)
                except:
                    self.train_iter = iter(self.train_dataloader)
                    batch = next(self.train_iter)
                losses, opdict = self.training_step(batch, step)
                if self.global_step % self.cfg.train.log_steps == 0:
                    loss_info = f"ExpName: {self.cfg.exp_name} \nEpoch: {epoch}, Iter: {step}/{iters_every_epoch}, Time: {datetime.now().strftime('%Y-%m-%d-%H:%M:%S')} \n"
                    for k, v in losses.items():
                        loss_info = loss_info + f'{k}: {v:.4f}, '
                        if self.cfg.train.write_summary:
                            self.writer.add_scalar('train_loss/'+k, v, global_step=self.global_step)                    
                    logger.info(loss_info)
                
                # if self.train_albedo and epoch%50==0:
                #     ckp_path = os.path.join('stylegan2', 'checkpoint')
                #     if not os.path.exists(ckp_path):
                #             os.mkdir(ckp_path)
                #     torch.save(
                #         {
                #             "g": self.albedoGAN.state_dict(),
                #         },
                #         # f"training_logs/checkpoint_z_1000/{str(i).zfill(6)}.pt",
                #         os.path.join(ckp_path, f'{str(epoch).zfill(6)}.pt'),
                #     )

                # if self.global_step % self.cfg.train.vis_steps == 0 and not self.train_albedo:
                if self.global_step % self.cfg.train.vis_steps == 0:
                    num_images = len(opdict['verts'])
                    visind = list(range(num_images))
                    # shape_images = self.deca.render.render_shape(opdict['verts'][visind], opdict['trans_verts'][visind])
                    render_faces = opdict['render_face'][visind]
                    pred_face_imgs = opdict['pred_face_img'][visind]
                    # pred_lm = opdict['pred_lmk'][visind].detach().cpu().numpy()
                    
                    visdict = {
                        'inputs': opdict['image'][visind], 
                        'gt_lmk_img': util.tensor_vis_landmarks(opdict['image'][visind], opdict['gt_lmk'][visind], isScale=True),
                        'pred_lmk_img': util.tensor_vis_landmarks(opdict['image'][visind], opdict['pred_lmk'][visind], isScale=True),
                        'render_faces': render_faces,
                        'pred_face_img': pred_face_imgs
                    }
                    # if 'predicted_images' in opdict.keys():
                    #     visdict['predicted_images'] = opdict['predicted_images'][visind]
                    # if 'predicted_detail_images' in opdict.keys():
                    #     visdict['predicted_detail_images'] = opdict['predicted_detail_images'][visind]

                    savepath = os.path.join(self.cfg.output_dir, self.cfg.train.vis_dir, f'{self.global_step:06}.jpg')
                    grid_image = util.visualize_grid(visdict, savepath, return_gird=True)
                    # import ipdb; ipdb.set_trace()                    
                    self.writer.add_image('train_images', (grid_image/255.).astype(np.float32).transpose(2,0,1), self.global_step)

                if self.global_step>0 and self.global_step % self.cfg.train.checkpoint_steps == 0:
                    model_dict = self.vifr.model_dict()
                    model_dict['opt'] = self.opt.state_dict()
                    model_dict['global_step'] = self.global_step
                    model_dict['batch_size'] = self.batch_size
                    torch.save(model_dict, os.path.join(self.cfg.output_dir, 'model' + '.tar'))   
                    # 
                    if self.global_step % self.cfg.train.checkpoint_steps*10 == 0:
                        os.makedirs(os.path.join(self.cfg.output_dir, 'models'), exist_ok=True)
                        torch.save(model_dict, os.path.join(self.cfg.output_dir, 'models', f'{self.global_step:08}.tar'))   

                # if self.global_step % self.cfg.train.val_steps == 0 and not self.train_albedo:
                if self.global_step % self.cfg.train.val_steps == 0:
                    # self.validation_step()
                    pass
                
                # if self.global_step % self.cfg.train.eval_steps == 0:
                #     self.evaluate()

                all_loss = losses['all_loss']
                self.opt.zero_grad(); all_loss.backward(); self.opt.step()
                self.global_step += 1
                if self.global_step > self.cfg.train.max_steps:
                    break