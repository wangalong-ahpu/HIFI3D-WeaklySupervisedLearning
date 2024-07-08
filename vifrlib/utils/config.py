'''
Default config for VIFR
'''
from yacs.config import CfgNode as CN
import argparse
import yaml
import os

cfg = CN()

abs_vifr_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
cfg.vifr_dir = abs_vifr_dir
cfg.device = 'cuda'
cfg.device_id = '0'

cfg.pretrained_modelpath = os.path.join(cfg.vifr_dir, 'data', 'deca_model.tar')
cfg.output_dir = os.path.join(cfg.vifr_dir, '../VIFR_TR')

cfg.checkpoints_dir = os.path.join(cfg.vifr_dir, 'checkpoints') # checkpoints总文件夹位置
cfg.topo_dir = os.path.join(cfg.vifr_dir, 'topo_assets') # topo总文件夹位置
cfg.rasterizer_type = 'pytorch3d'

# ---------------------------------------------------------------------------- #
# dataset数据集选项
# ---------------------------------------------------------------------------- #
cfg.dataset = CN()
cfg.dataset.training_data = ['facescape']
cfg.dataset.eval_data = ['aflw2000']
cfg.dataset.test_data = ['']
cfg.dataset.batch_size = 2
cfg.dataset.K = 1
cfg.dataset.isSingle = False
cfg.dataset.num_workers = 32
cfg.dataset.image_size = 224
cfg.dataset.scale_min = 1.4
cfg.dataset.scale_max = 1.8
cfg.dataset.trans_scale = 0.
cfg.dataset.facescape_dir = "/media/c723/SSD_1/Datasets/FaceScape/fmview_train"
cfg.dataset.small_facescape_dir = "/media/c723/SSD_1/Datasets/FaceScape/small_facescape"
cfg.dataset.sm_fs_dir = "/media/wang/SSD_1/Datasets/sm_fs"
cfg.dataset.vggface2_dir = "/media/c723/SSD_1/Datasets/vggface2"
cfg.dataset.lymh_dir = "/media/c723/SSD_1/Datasets/LYMH/image/1"
cfg.dataset.aflw2000_dir = "/media/c723/SSD_1/Datasets/aflw2000/image/1"


# ---------------------------------------------------------------------------- #
# model模型选项
# ---------------------------------------------------------------------------- #
cfg.model = CN()
cfg.model.fm_model_file = os.path.join(cfg.topo_dir, 'hifi3dpp_model_info.mat') # hifi3d++模型
cfg.model.unwrap_info_file = os.path.join(cfg.topo_dir, 'unwrap_1024_info.mat')
cfg.model.texgan_model_file = os.path.join(cfg.checkpoints_dir, 'texgan_model/texgan_ffhq_uv.pth') # texgan
cfg.model.net_recon_path = os.path.join(cfg.checkpoints_dir, 'deep3d_model/epoch_latest.pth') # deep3d预训练模型
cfg.model.vgg_model_path = os.path.join(cfg.checkpoints_dir, 'vgg_model/vgg16.pt') # vgg模型
cfg.model.E_hifi3d_backbone = 'resnet50' # 编码器主干
cfg.model.net_recog_path = os.path.join(cfg.checkpoints_dir, 'arcface_model/ms1mv3_arcface_r50_fp16_backbone.pth') # 人脸识别模型

cfg.model.net_recog = 'r50'
cfg.model.param_list = ['id', 'exp', 'tex', 'angle', 'gamma', 'trans'] # 参数列表
cfg.model.n_id = 532 # 形状系数
cfg.model.n_exp = 45 # 表情系数
cfg.model.n_tex = 439 # 纹理系数
cfg.model.n_angle = 3 # 角度
cfg.model.n_gamma = 27 # 光照参数
cfg.model.n_trans = 3 # 变换参数（暂时未知）
cfg.model.camera_distance = 10
cfg.model.focal = 1015.
cfg.model.center = 112.
cfg.model.znear = 5.
cfg.model.zfar = 15.
cfg.model.uselmk86 = False


# ---------------------------------------------------------------------------- #
# Losses 损失函数选项
# ---------------------------------------------------------------------------- #
cfg.loss = CN()
cfg.loss.lmk = 1.0
cfg.loss.mouthkc = 0.0
cfg.loss.useWlmk = True
cfg.loss.lmk_num = 68
cfg.loss.eyed = 1.0
cfg.loss.lipd = 0.5
cfg.loss.photo = 2.0
cfg.loss.vgg = 100.0

cfg.loss.useSeg = True
cfg.loss.mask_weight_dict_path = os.path.join(cfg.vifr_dir, 'data', 'mask_weight_dict.pkl')
cfg.loss.id = 0.2
cfg.loss.id_shape_only = True
cfg.loss.reg_id = 1e-04
cfg.loss.reg_exp = 1e-04
cfg.loss.reg_tex = 1e-04
cfg.loss.reg_gamma = 1.
cfg.loss.reg_jaw_pose = 0. #1.
cfg.loss.use_gender_prior = False
cfg.loss.shape_consistency = True
# loss for detail
cfg.loss.detail_consistency = True
cfg.loss.useConstraint = True
cfg.loss.mrf = 5e-2
cfg.loss.photo_D = 2.
cfg.loss.reg_sym = 0.005
cfg.loss.reg_z = 0.005
cfg.loss.reg_diff = 0.005

# ---------------------------------------------------------------------------- #
# train 训练选项
# ---------------------------------------------------------------------------- #
cfg.train = CN()
cfg.train.train_detail = False
cfg.train.train_albedo = False
cfg.train.max_epochs = 500000
cfg.train.max_steps = 1000000
cfg.train.lr = 1e-4
cfg.train.log_dir = 'logs'
cfg.train.log_steps = 10
cfg.train.vis_dir = 'train_images'
cfg.train.vis_steps = 10
cfg.train.write_summary = True
cfg.train.checkpoint_steps = 10
cfg.train.val_steps = 10
cfg.train.val_vis_dir = 'val_images'
cfg.train.eval_steps = 20
cfg.train.resume = True


















# # MICA Parameters
# cfg.use_mica = True
# cfg.train_other_params = False
# cfg.arcface_pretrained_model = '/scratch/is-rg-ncs/models_weights/arcface-torch/backbone100.pth'
# cfg.mapping_net_hidden_shape = 300
# cfg.mapping_layers = 3
# cfg.mica_model_path = os.path.join(cfg.deca_dir, 'data', 'mica_pretrained', 'mica.tar')


# cfg.pretrained_modelpath = os.path.join(cfg.deca_dir, 'data', 'deca_model.tar')
# cfg.output_dir = os.path.join(cfg.deca_dir, 'train')
# cfg.facescape_dir = "/media/c723/SSD_1/Datasets/FaceScape/fmview_train"
# cfg.small_facescape_dir = "/media/c723/SSD_1/Datasets/FaceScape/small_facescape"
# cfg.vggface2_dir = "/media/c723/SSD_1/Datasets/vggface2"
# cfg.lymh_dir = "/media/c723/SSD_1/Datasets/LYMH/image/1"
# cfg.aflw2000_dir = "/media/c723/SSD_1/Datasets/aflw2000/image/1"
# cfg.rasterizer_type = 'pytorch3d'
# # ---------------------------------------------------------------------------- #
# # Options for Face model
# # ---------------------------------------------------------------------------- #
# cfg.model = CN()
# cfg.model.topology_path = os.path.join(cfg.deca_dir, 'data', 'head_template.obj')
# # texture data original from http://files.is.tue.mpg.de/tbolkart/FLAME/FLAME_texture_data.zip
# cfg.model.dense_template_path = os.path.join(cfg.deca_dir, 'data', 'texture_data_256.npy')
# cfg.model.fixed_displacement_path = os.path.join(cfg.deca_dir, 'data', 'fixed_displacement_256.npy')
# cfg.model.flame_model_path = os.path.join(cfg.deca_dir, 'data', 'generic_model.pkl')
# cfg.model.E_flame_backbone = 'resnet50'  # resnet50 convnext_v1  convnext_v2_tiny  convnext_v2_atto
# cfg.model.flame_lmk_embedding_path = os.path.join(cfg.deca_dir, 'data', 'landmark_embedding.npy')
# cfg.model.flame_lmk191_indices_path = os.path.join(cfg.deca_dir, 'data', 'flame_lmk191_indices.npy')
# cfg.model.flame_lmk235_indices_path = os.path.join(cfg.deca_dir, 'data', 'flame_lmk235_indices.npy')
# cfg.model.face_mask_path = os.path.join(cfg.deca_dir, 'data', 'uv_face_mask.png')
# cfg.model.face_eye_mask_path = os.path.join(cfg.deca_dir, 'data', 'uv_face_eye_mask.png')
# cfg.model.mean_tex_path = os.path.join(cfg.deca_dir, 'data', 'mean_texture.jpg')
# cfg.model.tex_path = os.path.join(cfg.deca_dir, 'data', 'FLAME_albedo_from_BFM.npz')
# cfg.model.tex_type = 'BFM'  # BFM, FLAME, albedoMM
# cfg.model.uv_size = 256 #//////
# cfg.model.param_list = ['id', 'exp', 'tex', 'angle', 'gamma']
# cfg.model.n_id = 532 # 形状系数
# cfg.model.n_exp = 45 # 表情系数
# cfg.model.n_tex = 439 # 纹理系数
# cfg.model.n_angle = 3 # 角度
# cfg.model.n_gamma = 27 # 光照参数



# cfg.model.use_tex = True
# cfg.model.extract_tex = True
# cfg.model.jaw_type = 'aa' # default use axis angle, another option: euler. Note that: aa is not stable in the beginning
# # face recognition model
# cfg.model.fr_model_path = os.path.join(cfg.deca_dir, 'data', 'resnet50_ft_weight.pkl')

# ## details
# cfg.model.n_detail = 128
# cfg.model.max_z = 0.01

# # ---------------------------------------------------------------------------- #
# # Options for Dataset
# # ---------------------------------------------------------------------------- #
# cfg.dataset = CN()
# cfg.dataset.training_data = ['facescape']
# # cfg.dataset.training_data = ['ethnicity']
# cfg.dataset.eval_data = ['lymh']
# cfg.dataset.test_data = ['']
# cfg.dataset.batch_size = 2
# cfg.dataset.K = 1
# cfg.dataset.isSingle = False
# cfg.dataset.num_workers = 32
# cfg.dataset.image_size = 224
# cfg.dataset.scale_min = 1.4
# cfg.dataset.scale_max = 1.8
# cfg.dataset.trans_scale = 0.

# # ---------------------------------------------------------------------------- #
# # Options for training
# # ---------------------------------------------------------------------------- #
# cfg.train = CN()
# cfg.train.train_detail = False
# cfg.train.train_albedo = False
# cfg.train.max_epochs = 500000
# cfg.train.max_steps = 1000000
# cfg.train.lr = 1e-4
# cfg.train.log_dir = 'logs'
# cfg.train.log_steps = 10
# cfg.train.vis_dir = 'train_images'
# cfg.train.vis_steps = 10
# cfg.train.write_summary = True
# cfg.train.checkpoint_steps = 10
# cfg.train.val_steps = 10
# cfg.train.val_vis_dir = 'val_images'
# cfg.train.eval_steps = 20
# cfg.train.resume = True

# # ---------------------------------------------------------------------------- #
# # Options for Losses
# # ---------------------------------------------------------------------------- #
# cfg.loss = CN()
# cfg.loss.lmk = 1.0
# cfg.loss.mouthkc = 0.0  # 新新新新新新新新新新新新新新新新新新新新新新新新新新新新新新新新新新新新新新新
# cfg.loss.useWlmk = True
# cfg.loss.lmk_num = 68
# cfg.loss.eyed = 1.0
# cfg.loss.lipd = 0.5
# cfg.loss.photo = 2.0
# cfg.loss.useSeg = True
# cfg.loss.mask_weight_dict_path = os.path.join(cfg.deca_dir, 'data', 'mask_weight_dict.pkl')
# cfg.loss.id = 0.2
# cfg.loss.id_shape_only = True
# cfg.loss.reg_shape = 1e-04
# cfg.loss.reg_exp = 1e-04
# cfg.loss.reg_tex = 1e-04
# cfg.loss.reg_light = 1.
# cfg.loss.reg_jaw_pose = 0. #1.
# cfg.loss.use_gender_prior = False
# cfg.loss.shape_consistency = True
# # loss for detail
# cfg.loss.detail_consistency = True
# cfg.loss.useConstraint = True
# cfg.loss.mrf = 5e-2
# cfg.loss.photo_D = 2.
# cfg.loss.reg_sym = 0.005
# cfg.loss.reg_z = 0.005
# cfg.loss.reg_diff = 0.005


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return cfg.clone()

def update_cfg(cfg, cfg_file):
    cfg.merge_from_file(cfg_file)
    return cfg.clone()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='cfg file path')
    parser.add_argument('--mode', type=str, default='train', help='deca mode')

    args = parser.parse_args()
    print(args, end='\n')

    cfg = get_cfg_defaults()
    cfg.cfg_file = None
    cfg.mode = args.mode
    # import ipdb; ipdb.set_trace()
    if args.cfg is not None:
        cfg_file = args.cfg
        cfg = update_cfg(cfg, args.cfg)
        cfg.cfg_file = cfg_file

    return cfg
