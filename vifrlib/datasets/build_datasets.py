import os, sys
import torch
from torch.utils.data import Dataset, ConcatDataset
import torchvision.transforms as transforms
import numpy as np
import cv2
import scipy
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp, resize, rescale
from glob import glob

from .vggface import VGGFace2Dataset
from .ethnicity import EthnicityDataset
from .aflw2000 import AFLW2000, AFLW2000_T_Dataset
from .lymh import LYMH_TrainDataset, LYMH_TestDataset
from .facescape import Facescape_TrainDataset
from .small_facescape import SmallFacescapeDataset, SmallFaceScape_TrainDataset
from .now import NoWDataset
from .vox import VoxelDataset

def build_train(cfg, is_train=True):
    config = cfg.dataset
    data_list = []
    if 'vox2' in config.training_data:
        data_list.append(VoxelDataset(dataname='vox2', K=config.K, image_size=config.image_size, scale=[config.scale_min, config.scale_max], trans_scale=config.trans_scale, isSingle=config.isSingle))
    # if 'vggface2' in config.training_data:
    #     data_list.append(VGGFace2Dataset(K=config.K, image_size=config.image_size, scale=[config.scale_min, config.scale_max], trans_scale=config.trans_scale, isSingle=config.isSingle))
    if 'vggface2' in config.training_data:
        data_list.append(VGGFace2Dataset(K=config.K, image_size=config.image_size, scale=[config.scale_min, config.scale_max], trans_scale=config.trans_scale, \
                                         isSingle=config.isSingle, kpt_num=cfg.loss.lmk_num, train_path=cfg.dataset.vggface2_dir))
    if 'facescape' in config.training_data:
        data_list.append(Facescape_TrainDataset(K=config.K, image_size=config.image_size, scale=[config.scale_min, config.scale_max], trans_scale=config.trans_scale, \
                                                isSingle=config.isSingle, kpt_num=cfg.loss.lmk_num, train_path=cfg.dataset.facescape_dir))
    if 'small_facescape' in config.training_data:
        data_list.append(SmallFaceScape_TrainDataset(K=config.K, train_path=cfg.dataset.sm_fs_dir))
    if 'lymh' in config.training_data:
        data_list.append(LYMH_TrainDataset(K=config.K, image_size=config.image_size, scale=[config.scale_min, config.scale_max], trans_scale=config.trans_scale,\
                                           isSingle=config.isSingle, kpt_num=cfg.loss.lmk_num, train_path=cfg.dataset.lymh_dir))
    if 'aflw2000_train' in config.training_data:
        data_list.append(AFLW2000_T_Dataset(K=config.K, image_size=config.image_size, scale=[config.scale_min, config.scale_max], trans_scale=config.trans_scale, isSingle=config.isSingle))
    if 'vggface2hq' in config.training_data:
        data_list.append(VGGFace2HQDataset(K=config.K, image_size=config.image_size, scale=[config.scale_min, config.scale_max], trans_scale=config.trans_scale, isSingle=config.isSingle))
    if 'ethnicity' in config.training_data:
        data_list.append(EthnicityDataset(K=config.K, image_size=config.image_size, scale=[config.scale_min, config.scale_max], trans_scale=config.trans_scale, isSingle=config.isSingle))
    if 'coco' in config.training_data:
        data_list.append(COCODataset(image_size=config.image_size, scale=[config.scale_min, config.scale_max], trans_scale=config.trans_scale))
    if 'celebahq' in config.training_data:
        data_list.append(CelebAHQDataset(image_size=config.image_size, scale=[config.scale_min, config.scale_max], trans_scale=config.trans_scale))
    dataset = ConcatDataset(data_list)
    
    return dataset

def build_val(cfg, is_train=True):
    config = cfg.dataset
    data_list = []
    if 'vggface2' in config.eval_data:
        data_list.append(VGGFace2Dataset(isEval=True, K=config.K, image_size=config.image_size, scale=[config.scale_min, config.scale_max], trans_scale=config.trans_scale, isSingle=config.isSingle))
    if 'now' in config.eval_data:
        data_list.append(NoWDataset())
    if 'aflw2000' in config.eval_data:
        data_list.append(AFLW2000(testpath=cfg.dataset.aflw2000_dir))
    if 'lymh' in config.eval_data:
        data_list.append(LYMH_TestDataset(testpath=cfg.lymh_dir))
    dataset = ConcatDataset(data_list)

    return dataset
    