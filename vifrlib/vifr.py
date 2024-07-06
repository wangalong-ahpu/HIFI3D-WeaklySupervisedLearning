## This file has been taken from DECA and modified ##

import os, sys
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn

import numpy as np
from time import time
from skimage.io import imread
import cv2
# import pickle
# import matplotlib.pyplot as plt

# from .utils.renderer import SRenderY, set_rasterizer
from .models.encoders import ResnetEncoder
# from .models.FLAME import FLAME, FLAMETex
# from .models.decoders import Generator
from .utils import util
# from .utils.rotation_converter import batch_euler2axis
# from .utils.tensor_cropper import transform_points
# from .datasets import datasets
from .utils.config import cfg
# from .utils.util import load_obj
# from perlin_noise import rand_perlin_2d_octaves, rand_perlin_2d
from .models.texgan import TextureGAN
from .models.hifi3dpp import HIFI3D
# from img_2_tex import *
from .renderer_nvdiffrast import MeshRenderer
from .utils import lossfunc

torch.backends.cudnn.benchmark = True

class VIFR(nn.Module):
    def __init__(self, config=None, device='cuda:0'):
        super(VIFR, self).__init__()
        if config is None:
            self.cfg = cfg
        else:
            self.cfg = config
        self.device = device
        self.image_size = self.cfg.dataset.image_size
        # self.uv_size = self.cfg.model.uv_size
        # self.lmk_235_indices = torch.LongTensor(np.load(self.cfg.model.flame_lmk235_indices_path, allow_pickle=True)[0])
        # self.D_detail_flag = D_detail_flag
        # if self.D_detail_flag:
        #     self.D_detail = detail_GAN.to(device)

        # self.use_mica = use_mica
        # # if use_mica:
        #     # self.use_mica = use_mica
        # self.train_other_params = self.cfg.train_other_params

        self._create_model(self.cfg.model)
        self._setup_renderer(self.cfg.model)

        # if self.use_mica:
        #     self.app = FaceAnalysis(name='antelopev2', root='/media/wang/SSD_2/demo/AlbedoGAN/insightface/', providers=['CUDAExecutionProvider'])
        #     self.app.prepare(ctx_id=0, det_size=(224, 224))

    def _setup_renderer(self, model_cfg):
        # renderer
        fov = 2 * np.arctan(model_cfg.center / model_cfg.focal) * 180 / np.pi
        self.renderer = MeshRenderer(fov=fov, znear=model_cfg.znear, zfar=model_cfg.zfar, rasterize_size=int(2 * model_cfg.center))

        # # 原
        # set_rasterizer(self.cfg.rasterizer_type)
        # self.render = SRenderY(self.image_size, obj_filename=model_cfg.topology_path, uv_size=model_cfg.uv_size, rasterizer_type=self.cfg.rasterizer_type).to(self.device)
        # # face mask for rendering details
        # mask = imread(model_cfg.face_eye_mask_path).astype(np.float32)/255.; mask = torch.from_numpy(mask[:,:,0])[None,None,:,:].contiguous()
        # self.uv_face_eye_mask = F.interpolate(mask, [model_cfg.uv_size, model_cfg.uv_size]).to(self.device)
        # # self.uv_face_eye_mask = F.interpolate(mask, [256, 256]).to(self.device)
        # mask = imread(model_cfg.face_mask_path).astype(np.float32)/255.; mask = torch.from_numpy(mask[:,:,0])[None,None,:,:].contiguous()
        # self.uv_face_mask = F.interpolate(mask, [model_cfg.uv_size, model_cfg.uv_size]).to(self.device)
        # # displacement correction
        # fixed_dis = np.load(model_cfg.fixed_displacement_path)
        # self.fixed_uv_dis = torch.tensor(fixed_dis).float().to(self.device)
        # # mean texture
        # mean_texture = imread(model_cfg.mean_tex_path).astype(np.float32)/255.; mean_texture = torch.from_numpy(mean_texture.transpose(2,0,1))[None,:,:,:].contiguous()
        # self.mean_texture = F.interpolate(mean_texture, [model_cfg.uv_size, model_cfg.uv_size]).to(self.device)
        # # dense mesh template, for save detail mesh
        # self.dense_template = np.load(model_cfg.dense_template_path, allow_pickle=True, encoding='latin1').item()

    def _create_model(self, model_cfg):
        # 设置参数
        self.n_param = model_cfg.n_id+model_cfg.n_exp+model_cfg.n_tex+model_cfg.n_angle+model_cfg.n_gamma+model_cfg.n_trans
        # 数量列表
        self.num_list = [model_cfg.n_id, model_cfg.n_exp, model_cfg.n_tex, model_cfg.n_angle, model_cfg.n_gamma, model_cfg.n_trans]
        # 打印参数字典
        self.param_dict = {i:model_cfg.get('n_' + i) for i in model_cfg.param_list} # param_dict = {'id': 532, 'exp': 45, 'tex': 439, 'angle': 3, 'gamma': 27, 'trans':3}
        print(self.param_dict)


        # # mica models
        # if self.use_mica:
        #     arcface_pretrained_path = self.cfg.arcface_pretrained_model
        #     self.MICA_arcface = Arcface(pretrained_path=arcface_pretrained_path).to(self.device)
        #     arcface_embedding_dim = 512
        #     shape_output_dim = 300

        #     self.MICA_flameModel = MICAGenerator(arcface_embedding_dim,
        #                                             cfg.mapping_net_hidden_shape,
        #                                             shape_output_dim,
        #                                             cfg.mapping_layers,
        #                                             self.device)

        #     if self.train_other_params:
        #         # exp, pose, cam, light
        #         other_params_dim = 50 + 6 + 3 + 27
        #         self.MICA_otherParamModel = MICAGenerator(arcface_embedding_dim,
        #                                             cfg.mapping_net_hidden_shape,
        #                                             other_params_dim,
        #                                             cfg.mapping_layers,
        #                                             self.device)


        #     self.load_mica_model(self.cfg.mica_model_path)

        #     if self.train_other_params:
        #         self.MICA_otherParamModel.eval()

        #     self.MICA_arcface.eval()
        #     self.MICA_flameModel.eval()


        # encoders编码器
        if model_cfg.E_hifi3d_backbone == 'resnet50':
            self.E_hifi3d = ResnetEncoder(outsize=self.n_param).to(self.device)
        # if model_cfg.E_flame_backbone == 'convnext_v2_tiny':
        #     self.E_flame = Convnext_v2_T_Encoder(num_classes=self.n_param).to(self.device)
        # if model_cfg.E_flame_backbone == 'convnext_v2_atto':
        #     self.E_flame = Convnext_v2_A_Encoder(num_classes=self.n_param).to(self.device)
        # self.E_detail = ResnetEncoder(outsize=self.n_detail).to(self.device)
        
        # decoders解码器
        ## hifi3d++ 人脸参数化模型
        self.hifi3dpp = HIFI3D(fm_model_file=model_cfg.fm_model_file,
                                                    unwrap_info_file=model_cfg.unwrap_info_file,
                                                    camera_distance=model_cfg.camera_distance,
                                                    focal=model_cfg.focal,
                                                    center=model_cfg.center,
                                                    lm86=False,
                                                    device=self.device)

        # texture gan 纹理GAN
        self.tex_gan = TextureGAN(model_path=self.cfg.model.texgan_model_file, device=self.device)

        # resume model 重开模型
        model_path = self.cfg.pretrained_modelpath
        if os.path.exists(model_path):
            print(f'trained model found. load {model_path}')
            checkpoint = torch.load(model_path)
            self.checkpoint = checkpoint
            util.copy_state_dict(self.E_hifi3d.state_dict(), checkpoint['E_hifi3d'])
            # util.copy_state_dict(self.E_detail.state_dict(), checkpoint['E_detail'])

            # if self.D_detail_flag is False:
            #     util.copy_state_dict(self.D_detail.state_dict(), checkpoint['D_detail'])

        else:
            print(f'please check model path: {model_path}')
            # exit()
        # eval mode
        self.E_hifi3d.eval()
        # self.E_detail.eval()
        # self.D_detail.eval()



    def load_mica_model(self, model_path):
        if os.path.exists(model_path):
            print(f'Trained model found. Path: {model_path} | GPU: {self.device}')
            checkpoint = torch.load(model_path)
            if 'arcface' in checkpoint:
                self.MICA_arcface.load_state_dict(checkpoint['arcface'])
            if 'flameModel' in checkpoint:
                self_state_dict = self.MICA_flameModel.state_dict()
                for name, param in checkpoint['flameModel'].items():
                    if name not in self_state_dict:
                        continue

                    if isinstance(param, torch.nn.parameter.Parameter):
                        param = param.data

                    self_state_dict[name].copy_(param)

            if self.train_other_params and 'otherParamModel' in checkpoint:
                self_state_dict = self.MICA_otherParamModel.state_dict()
                for name, param in checkpoint['otherParamModel'].items():
                    if name not in self_state_dict:
                        continue

                    if isinstance(param, torch.nn.parameter.Parameter):
                        param = param.data

                    self_state_dict[name].copy_(param)
        else:
            print(f'Checkpoint not available starting from scratch!')



    def decompose_code(self, code, num_dict):
        ''' 
        分解系数
        param_dict = {'id': 532, 'exp': 45, 'tex': 439, 'angle': 3, 'gamma': 27, 'trans':3}
        '''
        code_dict = {}
        start = 0
        for key in num_dict:
            end = start+int(num_dict[key])
            code_dict[key] = code[:, start:end]
            start = end
            if key == 'gamma':
                code_dict[key] = code_dict[key].reshape(code_dict[key].shape[0], 9, 3)
        return code_dict

    def displacement2normal(self, uv_z, coarse_verts, coarse_normals):
        ''' Convert displacement map into detail normal map
        '''
        batch_size = uv_z.shape[0]
        uv_coarse_vertices = self.render.world2uv(coarse_verts).detach()
        uv_coarse_normals = self.render.world2uv(coarse_normals).detach()

        self.uv_face_eye_mask = torchvision.transforms.Resize(uv_z.shape[3])(self.uv_face_eye_mask)
        uv_coarse_vertices = torchvision.transforms.Resize(uv_z.shape[3])(uv_coarse_vertices)
        uv_coarse_normals = torchvision.transforms.Resize(uv_z.shape[3])(uv_coarse_normals)
        self.fixed_uv_dis = torchvision.transforms.Resize(uv_z.shape[3])(self.fixed_uv_dis[None,None,:,:])
        self.fixed_uv_dis = self.fixed_uv_dis.squeeze(0).squeeze(0)

    
        uv_z = uv_z*self.uv_face_eye_mask
        uv_detail_vertices = uv_coarse_vertices + uv_z*uv_coarse_normals + self.fixed_uv_dis[None,None,:,:]*uv_coarse_normals.detach()
        dense_vertices = uv_detail_vertices.permute(0,2,3,1).reshape([batch_size, -1, 3])
        uv_detail_normals = util.vertex_normals(dense_vertices, self.render.dense_faces.expand(batch_size, -1, -1))
        uv_detail_normals = uv_detail_normals.reshape([batch_size, uv_coarse_vertices.shape[2], uv_coarse_vertices.shape[3], 3]).permute(0,3,1,2)
        uv_detail_normals = uv_detail_normals*self.uv_face_eye_mask + uv_coarse_normals*(1.-self.uv_face_eye_mask)
        return uv_detail_normals

    def visofp(self, normals):
        ''' visibility of keypoints, based on the normal direction
        '''
        normals68 = self.flame.seletec_3d68(normals)
        vis68 = (normals68[:,:,2:] < 0.1).float()
        return vis68

    def encode(self, image):
        """
        step1:预测
        step2:分解
        """
        # 图像->预测系数
        pred_coeffs  = self.E_hifi3d(image)
        # 预测系数分解
        codedict = self.decompose_code(pred_coeffs, self.param_dict)
        codedict['image'] = image
        return codedict
    
    def infer_render(self, is_uv_tex=True):
        # forward face model
        self.pred_coeffs_dict = self.facemodel.split_coeff(self.pred_coeffs)
        self.pred_vertex, self.pred_tex, self.pred_shading, self.pred_color, self.pred_lm = \
            self.facemodel.compute_for_render(codedict)
        if is_uv_tex:
            # forward texture gan
            self.pred_uv_map = self.tex_gan.synth_uv_map(self.pred_latents_w)
            # render front face
            vertex_uv_coord = self.facemodel.vtx_vt.unsqueeze(0).repeat(self.pred_coeffs.size()[0], 1, 1)
            render_feat = torch.cat([vertex_uv_coord, self.pred_shading], axis=2)
            self.render_face_mask, _, self.render_face = \
                self.renderer(self.pred_vertex, self.facemodel.face_buf, feat=render_feat, uv_map=self.pred_uv_map)
        else:
            # render front face
            self.render_face_mask, _, self.render_face = \
                self.renderer(self.pred_vertex, self.facemodel.face_buf, feat=self.pred_color)
            
    def decode(self, codedict, is_uv_tex=False, name="None", rendering=True, iddict=None, vis_lmk=True, return_vis=True, use_detail=True,
                render_orig=False, original_image=None, tform=None):
        """
        infer_render
        """
        image = codedict['image']
        batch_size = image.shape[0]

        # 参数->hifi3d拓扑人脸模型
        pred_vertex, pred_tex, pred_shading, pred_color, pred_lmk = self.hifi3dpp.compute_for_render(codedict)
        pred_lmk[..., 1] = self.cfg.dataset.image_size - 1 - pred_lmk[..., 1]
        pred_lmk = pred_lmk/(self.cfg.dataset.image_size/2)-1
        if is_uv_tex:
            # forward texture gan
            pred_uv_map = self.tex_gan.synth_uv_map(self.pred_latents_w)
            # render front face
            vertex_uv_coord = self.hifi3dpp.vtx_vt.unsqueeze(0).repeat(codedict.size()[0], 1, 1)
            render_feat = torch.cat([vertex_uv_coord, self.pred_shading], axis=2)
            # self.render_face_mask, _, self.render_face = self.renderer(pred_vertex, self.hifi3dpp.face_buf, feat=render_feat, uv_map=pred_uv_map)
            render_face_mask, _, render_face = self.renderer(pred_vertex, self.hifi3dpp.face_buf, feat=render_feat, uv_map=pred_uv_map)
        else:
            # render front face
            render_face_mask, _, render_face = self.renderer(pred_vertex, self.hifi3dpp.face_buf, feat=pred_color)
        
        pred_face_img = render_face * render_face_mask + (1 - render_face_mask) * image
        # pred_face_img = util.tensor2np(pred_face_img)
        x="debug"
        opdict = {
                'verts': pred_vertex,
                'texs': pred_tex,
                'pred_shading': pred_shading,
                'pred_color': pred_color,
                'pred_lmk': pred_lmk,
                'render_face_mask': render_face_mask,
                'render_face': render_face,
                'pred_face_img': pred_face_img
            }
        return opdict

        # self.infer_render(is_uv_tex=False)
        # vis_img = self.visualize(input_data, is_uv_tex=False)
        # vis_tex_uv = self.visualize_3dmmtex_as_uv()
        # logger.write_disk_images([vis_img], ['stage1_vis'])
        # logger.write_disk_images([vis_tex_uv], ['stage1_vis_3dmmtex_as_uv'])
        # self.save_mesh(path=logger.vis_dir, mesh_name='stage1_mesh.obj', is_uv_tex=False)
        # self.save_coeffs(path=logger.vis_dir, coeffs_name='stage1_coeffs.pt', is_uv_tex=False)
        # verts, landmarks2d, landmarks3d = self.flame(shape_params=shape_params, expression_params=codedict['exp'], pose_params=codedict['pose'])
        # 参数->hifi3d拓扑人脸模型
        # pred_vertex, pred_tex, pred_shading, pred_color, pred_lm = self.facemodel.compute_for_render(self.pred_coeffs_dict)

        # if self.cfg.model.use_tex:
        #     albedo = self.flametex(codedict['tex'])
        # else:
        #     albedo = torch.zeros([batch_size, 3, self.uv_size, self.uv_size], device=images.device) 
        # landmarks3d_world = landmarks3d.clone()

        # ## projection
        # landmarks2d = util.batch_orth_proj(landmarks2d, codedict['cam'])[:,:,:2]; landmarks2d[:,:,1:] = -landmarks2d[:,:,1:]#; landmarks2d = landmarks2d*self.image_size/2 + self.image_size/2
        # landmarks3d = util.batch_orth_proj(landmarks3d, codedict['cam']); landmarks3d[:,:,1:] = -landmarks3d[:,:,1:] #; landmarks3d = landmarks3d*self.image_size/2 + self.image_size/2
        # trans_verts = util.batch_orth_proj(verts, codedict['cam']); trans_verts[:,:,1:] = -trans_verts[:,:,1:]
        # # landmarks2d_191 = trans_verts[:, self.lmk_191_indices, :2]
        # landmarks2d_235 = trans_verts[:, self.lmk_235_indices, :2]
        # # id_loss = lossfunc.VGGFace2Loss(pretrained_model=self.cfg.model.fr_model_path)



        
        # # dense_trans_verts = util.batch_orth_proj(dense_verts, codedict['cam']); dense_trans_verts[:,:,1:] = -dense_trans_verts[:,:,1:]
        # opdict = {
        #     'verts': verts,
        #     'trans_verts': trans_verts,
        #     'landmarks2d': landmarks2d,
        #     # 'landmarks2d_191': landmarks2d_191,
        #     'landmarks2d_235': landmarks2d_235,
        #     'landmarks3d': landmarks3d,
        #     'landmarks3d_world': landmarks3d_world,
        # }

        # ## rendering
        # if return_vis and render_orig and original_image is not None and tform is not None:
        #     points_scale = [self.image_size, self.image_size]
        #     _, _, h, w = original_image.shape
        #     # import ipdb; ipdb.set_trace()
        #     trans_verts = transform_points(trans_verts, tform, points_scale, [h, w])    # points_scale [224, 224], [h,w] = [1024, 1024]
        #     # dense_trans_verts = transform_points(dense_trans_verts, tform, points_scale, [h, w]) 
        #     landmarks2d = transform_points(landmarks2d, tform, points_scale, [h, w])
        #     landmarks3d = transform_points(landmarks3d, tform, points_scale, [h, w])
        #     background = original_image
        #     images = original_image
        # else:
        #     h, w = self.image_size, self.image_size
        #     background = None

        # if rendering:

        #     ops = self.render(verts, trans_verts, albedo, h=h, w=w, background=background)
        #     ## output

        #     opdict['grid'] = ops['grid']
        #     opdict['rendered_images'] = ops['images']
        #     opdict['alpha_images'] = ops['alpha_images']
        #     opdict['normal_images'] = ops['normal_images']
        
        # if self.cfg.model.use_tex:
        #     opdict['albedo'] = albedo
            
        # if use_detail:
        #     rend_size = 256

        #     if self.D_detail_flag:
        #         uv_z, _ = self.D_detail([codedict['latent']], input_is_latent=True)
        #         uv_z = uv_z[:,:1,:,:] 
        #         uv_z = uv_z.clip(0,1)
        #         uv_z = (2*uv_z-1)/255
        #         uv_z = uv_z.clip(-0.004, 0.004)

        #     else:
        #         uv_z = self.D_detail(torch.cat([codedict['pose'][:,3:], codedict['exp'], codedict['detail']], dim=1))   # torch.Size([1, 1, 256, 256])
        #         uv_z = torchvision.transforms.Resize(rend_size)(uv_z)


        #     # noise = rand_perlin_2d_octaves((rend_size, rend_size), (8, 8), 5)
        #     # uv_z = torchvision.transforms.Resize(rend_size)(uv_z)
        #     # noise = rand_perlin_2d_octaves((rend_size, rend_size), (8, 8), 5)
        #     # noise = rand_perlin_2d((rend_size, rend_size), (8, 8))

        #     # uv_z = uv_z + noise.to('cuda')*0.00

        #     # if iddict is not None:
        #         # uv_z = self.D_detail(torch.cat([iddict['pose'][:,3:], iddict['exp'], codedict['detail']], dim=1))

        #     uv_detail_normals = self.displacement2normal(uv_z, verts, ops['normals'])
        #     uv_shading = self.render.add_SHlight(uv_detail_normals, codedict['light'])
        #     albedo = torchvision.transforms.Resize(uv_shading.shape[3])(albedo)
        #     uv_texture = albedo*uv_shading

        #     opdict['uv_texture'] = uv_texture 
        #     opdict['normals'] = ops['normals']
        #     opdict['uv_detail_normals'] = uv_detail_normals
        #     opdict['displacement_map'] = uv_z+self.fixed_uv_dis[None,None,:,:]
        
        # if vis_lmk:
        #     landmarks3d_vis = self.visofp(ops['transformed_normals'])#/self.image_size
        #     landmarks3d = torch.cat([landmarks3d, landmarks3d_vis], dim=2)
        #     opdict['landmarks3d'] = landmarks3d

        # if return_vis:
        #     ## render shape
        #     if background is None:
        #         background = torch.ones(1,3,224,224).to('cuda')
        #     shape_images, _, grid, alpha_images = self.render.render_shape(verts, trans_verts, h=h, w=w, images=background, return_grid=True)
        #     detail_normal_images = F.grid_sample(uv_detail_normals, grid, align_corners=False)*alpha_images
        #     shape_detail_images = self.render.render_shape(verts, trans_verts, detail_normal_images=detail_normal_images, h=h, w=w, images=background)
            
        #     ## extract texture
        #     # uv_pverts = self.render.world2uv(trans_verts)
        #     # uv_pverts2 = torchvision.transforms.Resize(1024)(uv_pverts)

        #     rend_size = 1024

        #     uv_pverts_2 = self.render.world2uv_custom(trans_verts, tex_size=rend_size, faces=None, uvcoords=None, uvfaces=None)
        #     # uv_pverts_dense = self.render.world2uv_custom(
        #     #                                                 dense_trans_verts.to(device), 
        #     #                                                 tex_size=1024, 
        #     #                                                 faces=dense_faces.to(device), 
        #     #                                                 uvcoords=dense_uvcoords_padded.to(device), 
        #     #                                                 uvfaces=dense_uv_faces.to(device)
        #     #                                                 )

        #     uv_gt = F.grid_sample(images, uv_pverts_2.permute(0,2,3,1)[:,:,:,:2], mode='bilinear', align_corners=False)

        #     background = images
            
        #     # from PIL import Image
        #     # uv_gt = Image.open('')

        #     ###### Add Perlin Noise to Texture

        #     noise = rand_perlin_2d_octaves((2048, 2048), (128, 128), 5)
        #     uv_texture = torchvision.transforms.Resize(rend_size)(uv_texture)[:,:3,:,:] + noise[None,None,:1024,:1024].to('cuda')/50

        #     uv_texture_gt = torchvision.transforms.Resize(rend_size)(uv_gt[:,:3,:,:])*torchvision.transforms.Resize(rend_size)(self.uv_face_eye_mask) + \
        #                     (torchvision.transforms.Resize(rend_size)(uv_texture)[:,:3,:,:]* \
        #                     (1-torchvision.transforms.Resize(rend_size)(self.uv_face_eye_mask)))

        #     ### Texture Correction

        #     '''angle1 = mesh_angle(verts[0].detach().cpu().numpy(), [3572,3555,2205])
        #     angle2 = mesh_angle(verts[0].detach().cpu().numpy(), [3572,723,3555])
        #     avg_ang = int((angle1+angle2)/2)
        #     avg_ang = 90-(360-avg_ang)
        #     correct_tex = tex_correction(uv_texture_gt[0].permute(1,2,0).detach().cpu(), avg_ang)
        #     uv_texture_gt = correct_tex.permute(2,0,1)[None,...].to('cuda')'''

        #     # uv_shading = self.render.add_SHlight(uv_detail_normals, codedict['light'])


        #     white_bg = torch.ones_like(images)
        #     ## output
        #     self.create_sh_video=0

        #     if self.create_sh_video:

        #         print('creating SH video...')
        #         size = (1024,1024)
        #         out_path = 'paper_samples/disp_gan/videos_sh/'
        #         # img_no = '0000ab'
        #         sh_coeff1 = torch.from_numpy(np.load('sh_light_1.npy')).to('cuda')
        #         sh_coeff2 = torch.from_numpy(np.load('sh_light_2.npy')).to('cuda')
        #         sh_coefs = codedict['light']

        #         # uv_shading2 = self.render.add_SHlight(uv_detail_normals, sh_coeff2)
        #         uv_shading1 = self.render.add_SHlight(uv_detail_normals, sh_coeff1)
        #         uv_shading = self.render.add_SHlight(uv_detail_normals, sh_coefs)

        #         out = cv2.VideoWriter(os.path.join(out_path, name+'.mp4'), cv2.VideoWriter_fourcc(*'MP4V'), 90, size)
                

        #         # for angle in range(0,1000):

        #         #     # sh_coefs[:,5,:] = codedict['light'][:,5,:] + angle/10000
        #         #     sh_coefs = sh_coeff1*(1-angle/100) + sh_coeff2*(angle/100)
        #         #     ops = self.render(verts, trans_verts, uv_texture_gt, lights=sh_coefs, h=rend_size, w=rend_size, background=white_bg)
        #         #     out.write(cv2.cvtColor(((ops['images'][0].permute(1,2,0)*255).clip(0,255))[:,:,:3].to(torch.uint8).cpu().numpy(), cv2.COLOR_BGR2RGB))

        #         for i in range(191):
        #             uv_sh = uv_shading1*(1-i/100) + uv_shading*(i/100)*1.1
        #             ops = self.render(verts, trans_verts, uv_texture_gt*torchvision.transforms.Resize(1024)(uv_sh), h=rend_size, w=rend_size, background=white_bg)
        #             out.write(cv2.cvtColor(((ops['images'][0].permute(1,2,0)*255).clip(0,255))[:,:,:3].to(torch.uint8).cpu().numpy(), cv2.COLOR_BGR2RGB))
        #             # print(i, ops['images'][0].min())
        #             if ops['images'][0].min()==1:
        #                 break
        #             # if i==190:
        #             #     break
        #             #cv2.imshow('shading', ops['images'][0].permute(1,2,0).detach().cpu().numpy())
        #             #cv2.waitKey(1)
        #         # out.release()
                
        #         # for angle in range(-200,0):

        #         #     sh_coefs[:,4,:] = codedict['light'][:,4,:] + angle/10000
        #         #     # sh_coefs = sh_coeff1*(1-angle/1000) + sh_coeff2*(angle/1000)
        #         #     ops = self.render(verts, trans_verts, uv_texture_gt, lights=sh_coefs, h=rend_size, w=rend_size, background=white_bg)
        #         #     out.write(cv2.cvtColor(((ops['images'][0].permute(1,2,0)*255).clip(0,255))[:,:,:3].to(torch.uint8).cpu().numpy(), cv2.COLOR_BGR2RGB))

        #         out.release()

        #         # out = cv2.VideoWriter(os.path.join(out_path, img_no+'.mp4'), cv2.VideoWriter_fourcc(*'MP4V'), 90, size)

        #         # for i in range(1000):
        #         #     uv_sh = uv_shading1*2*(i/100) + uv_shading*2*(1-i/100)
        #         #     ops = self.render(verts, trans_verts, uv_texture_gt*torchvision.transforms.Resize(1024)(uv_sh), h=rend_size, w=rend_size, background=white_bg)
        #         #     out.write(cv2.cvtColor(((ops['images'][0].permute(1,2,0)*255).clip(0,255))[:,:,:3].to(torch.uint8).cpu().numpy(), cv2.COLOR_BGR2RGB))
        #         #     print(i, ops['images'][0].min())
        #         #     if ops['images'][0].min()==1:
        #         #         break
        #         #     if i==191:
        #         #         break
        #         #     #cv2.imshow('shading', ops['images'][0].permute(1,2,0).detach().cpu().numpy())
        #         #     #cv2.waitKey(1)
        #         # out.release()



        #     else:
        #         size = (1024,1024)
        #         ops = self.render(verts, trans_verts, uv_texture_gt, h=rend_size, w=rend_size, background=white_bg)


        #     opdict['grid'] = ops['grid']
        #     opdict['rendered_images'] = ops['images']
        #     opdict['alpha_images'] = ops['alpha_images']
        #     opdict['normal_images'] = ops['normal_images']


        #     if self.cfg.model.use_tex:
        #         ## TODO: poisson blending should give better-looking results
        #         if self.cfg.model.extract_tex:
        #             self.uv_face_eye_mask = torchvision.transforms.Resize(rend_size)(self.uv_face_eye_mask)
        #             uv_texture = torchvision.transforms.Resize(rend_size)(uv_texture)
        #             uv_texture_gt = uv_gt[:,:3,:,:]*self.uv_face_eye_mask + (uv_texture[:,:3,:,:]*(1-self.uv_face_eye_mask))

        #             ### Texture Correction

        #             '''angle1 = mesh_angle(verts[0].detach().cpu().numpy(), [3572,3555,2205])
        #             angle2 = mesh_angle(verts[0].detach().cpu().numpy(), [3572,723,3555])
        #             avg_ang = int((angle1+angle2)/2)
        #             avg_ang = 90-(360-avg_ang)
        #             correct_tex = tex_correction(uv_texture_gt[0].permute(1,2,0).detach().cpu(), avg_ang)
        #             uv_texture_gt = correct_tex.permute(2,0,1)[None,...].to('cuda')
        #             uv_texture_gt = uv_texture_gt[:,:3,:,:]*self.uv_face_eye_mask + (uv_texture[:,:3,:,:]*(1-self.uv_face_eye_mask))'''


        #         else:
        #             uv_texture_gt = uv_texture[:,:3,:,:]
        #     else:
        #         uv_texture_gt = uv_gt[:,:3,:,:]*self.uv_face_eye_mask + (torch.ones_like(uv_gt[:,:3,:,:])*(1-self.uv_face_eye_mask)*0.7)
                
        #     landmarks2d_img = util.tensor_vis_landmarks(images, landmarks2d)
        #     landmarks3d_img = util.tensor_vis_landmarks(images, landmarks3d)
            
        #     opdict['uv_texture_gt'] = uv_texture_gt
        #     visdict = {
        #         'inputs': images, 
        #         'landmarks2d': landmarks2d_img,
        #         'landmarks3d': landmarks3d_img,
        #         'shape_images': shape_images,
        #         'shape_detail_images': shape_detail_images
        #     }
        #     if self.cfg.model.use_tex:
        #         visdict['rendered_images'] = ops['images']

        #     return opdict, visdict

        # else:
        #     return opdict

    def visualize(self, visdict, size=224, dim=2):
        '''
        image range should be [0,1]
        dim: 2 for horizontal. 1 for vertical
        '''
        assert dim == 1 or dim==2
        grids = {}
        for key in visdict:
            _,_,h,w = visdict[key].shape
            if dim == 2:
                new_h = size; new_w = int(w*size/h)
            elif dim == 1:
                new_h = int(h*size/w); new_w = size
            grids[key] = torchvision.utils.make_grid(F.interpolate(visdict[key], [new_h, new_w]).detach().cpu())
        grid = torch.cat(list(grids.values()), dim)
        grid_image = (grid.numpy().transpose(1,2,0).copy()*255)[:,:,[2,1,0]]
        grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)
        return grid_image
    
    def save_obj(self, filename, opdict, codedict):
        '''
        vertices: [nv, 3], tensor
        texture: [3, h, w], tensor
        '''
        i = 0
        vertices = opdict['verts'][i].cpu().numpy()
        faces = self.render.faces[0].cpu().numpy()
        texture = util.tensor2image(opdict['uv_texture_gt'][i])
        uvcoords = self.render.raw_uvcoords[0].cpu().numpy()
        uvfaces = self.render.uvfaces[0].cpu().numpy()
        # save coarse mesh, with texture and normal map
        normal_map = util.tensor2image(opdict['uv_detail_normals'][i]*0.5 + 0.5)
        # util.write_obj(filename, vertices, faces, 
        #                 texture=texture, 
        #                 uvcoords=uvcoords, 
        #                 uvfaces=uvfaces, 
        #                 normal_map=normal_map)
        # upsample mesh, save detailed mesh
        
        texture = texture[:,:,[2,1,0]]
        normals = opdict['normals'][i].cpu().numpy()
        displacement_map = opdict['displacement_map'][i].cpu().numpy().squeeze()
        dense_vertices, dense_colors, dense_faces, dense_uv_coords = util.upsample_mesh(vertices, normals, faces, displacement_map, texture, self.dense_template)
        # util.write_obj(filename.replace('.obj', '_detail.obj'), 
        util.write_obj(filename,
                        dense_vertices,           # (59315, 3)
                        dense_faces,              # (117380, 3)
                        texture=texture, 
                        uvcoords=dense_uv_coords, 
                        uvfaces=dense_faces)
    
    
    def run(self, imagepath, iscrop=True):
        ''' 
        An api for running deca given an image path
        '''
        testdata = datasets.TestData(imagepath)
        images = testdata[0]['image'].to(self.device)[None,...]
        codedict = self.encode(images)
        opdict, visdict = self.decode(codedict)
        return codedict, opdict, visdict

    def model_dict(self):
        # return {
        #     'E_flame': self.E_flame.state_dict(),
        #     'E_detail': self.E_detail.state_dict(),
        #     'D_detail': self.D_detail.state_dict(),
        #     'flameModel': self.MICA_flameModel.state_dict(),
        #     'otherParamModel': self.MICA_otherParamModel.state_dict(),
        #     'arcface': self.MICA_arcface.state_dict()
        # }
        return {
                'E_flame': self.E_hifi3d.state_dict(),
                # 'E_detail': self.E_detail.state_dict(),
                # 'D_detail': self.D_detail.state_dict()
            }


    def decode_tex(self, codedict, rendering=True, iddict=None, vis_lmk=True, return_vis=True, use_detail=True,
                render_orig=False, original_image=None, tform=None):

        images = codedict['images']
        batch_size = images.shape[0]
        
        ## decode
        verts, landmarks2d, landmarks3d = self.flame(shape_params=codedict['shape'], expression_params=codedict['exp'], pose_params=codedict['pose'])
        uv_texture = self.flametex(codedict['tex'])

        # landmarks3d_world = landmarks3d.clone()

        ## projection
        landmarks2d = util.batch_orth_proj(landmarks2d, codedict['cam'])[:,:,:2]; landmarks2d[:,:,1:] = -landmarks2d[:,:,1:]
        landmarks3d = util.batch_orth_proj(landmarks3d, codedict['cam']); landmarks3d[:,:,1:] = -landmarks3d[:,:,1:]
        trans_verts = util.batch_orth_proj(verts, codedict['cam']); trans_verts[:,:,1:] = -trans_verts[:,:,1:]

        ## rendering
        if return_vis and render_orig and original_image is not None and tform is not None:
            
            points_scale = [self.image_size, self.image_size]
            _, _, h, w = original_image.shape
            trans_verts = transform_points(trans_verts, tform, points_scale, [h, w])    # points_scale [224, 224], [h,w] = [1024, 1024]
            dense_trans_verts = transform_points(dense_trans_verts, tform, points_scale, [h, w]) 
            landmarks2d = transform_points(landmarks2d, tform, points_scale, [h, w])
            landmarks3d = transform_points(landmarks3d, tform, points_scale, [h, w])
            background = original_image
            images = original_image

        else:
            h, w = self.image_size, self.image_size
            background = None
            

        if return_vis:
            # shape_images, _, grid, alpha_images = self.render.render_shape(verts, trans_verts, h=h, w=w, images=background, return_grid=True)

            uv_pverts_2 = self.render.world2uv_custom(trans_verts, tex_size=1024, faces=None, uvcoords=None, uvfaces=None)

            uv_gt = F.grid_sample(images, uv_pverts_2.permute(0,2,3,1)[:,:,:,:2], mode='bilinear', align_corners=False)

            background = images
            h, w = 1024, 1024
            # from PIL import Image
            uv_texture_gt = uv_gt[:,:3,:,:]*torchvision.transforms.Resize(1024)(self.uv_face_eye_mask) + \
                                    (torchvision.transforms.Resize(1024)(uv_texture)[:,:3,:,:]*(1-torchvision.transforms.Resize(1024)(self.uv_face_eye_mask)))
            # ops = self.render(verts, trans_verts, uv_texture_gt, h=h, w=w, background=None)

            ## TODO: try poisson blending 

            ## Adding Perlin Noise to Texture

            noise = rand_perlin_2d_octaves((2048, 2048), (128, 128), 5)
            uv_texture = torchvision.transforms.Resize(h)(uv_texture)[:,:3,:,:] + noise[None,None,:1024,:1024].to('cuda')/50

            self.uv_face_eye_mask = torchvision.transforms.Resize(1024)(self.uv_face_eye_mask)
            uv_texture = torchvision.transforms.Resize(1024)(uv_texture)
            uv_texture_gt = uv_gt[:,:3,:,:]*self.uv_face_eye_mask + (uv_texture[:,:3,:,:]*(1-self.uv_face_eye_mask))
            

        return uv_texture_gt, verts, self.uv_face_eye_mask, uv_texture


# if __name__ == '__main__':
#     from vifrlib.utils.config import parse_args
#     cfg = parse_args()
#     if cfg.cfg_file is not None:
#         exp_name = cfg.cfg_file.split('/')[-1].split('.')[0]  # deca_pretrain or deca_coarse
#         cfg.exp_name = exp_name
#     vifr = VIFR(cfg)