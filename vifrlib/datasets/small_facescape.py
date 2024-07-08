import os, sys
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import scipy
import pickle
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp, resize, rescale
from glob import glob
from torch.utils.data import Dataset, DataLoader, ConcatDataset


class SmallFacescapeDataset(Dataset):
    def __init__(self, K, image_size, scale, trans_scale=0, isTemporal=False, isEval=False, isSingle=False, kpt_num=235,
                 train_path=None):
        '''
        K must be less than 6
        '''
        self.K = K
        self.image_size = image_size
        self.imagefolder = os.path.join(train_path, 'image/')
        self.kptfolder = os.path.join(train_path, 'kpt' + str(kpt_num) + '/')
        self.segfolder = os.path.join(train_path, 'seg_fp' + '/')
        self.arcfacefolder = os.path.join(train_path, 'arcface' + '/')
        with open("/media/wang/SSD_1/demo/VIFR/data/mask_weight_dict.pkl", 'rb') as f:
            self.mask_weight_dict = pickle.load(f)

        hou = '.npy' if self.K == 1 else '_K.npy'
        datafile = os.path.join(train_path, 'train_list_kpt' + str(kpt_num) + hou)
        if isEval:
            datafile = os.path.join(train_path, 'train_list_kpt' + str(kpt_num) + hou)

        # datafile = os.path.join(train_path, 'train_list_kpt' + str(kpt_num) + '_new.npy')
        # if isEval:
        #     datafile = os.path.join(train_path, 'train_list_kpt' + str(kpt_num) + '_new.npy')

        self.data_lines = np.load(datafile).astype('str')

        self.isTemporal = isTemporal
        self.scale = scale  # [scale_min, scale_max]
        self.trans_scale = trans_scale  # [dx, dy]
        self.isSingle = isSingle
        if isSingle:
            self.K = 1

    def __len__(self):
        return len(self.data_lines)

    def __getitem__(self, idx):
        images_list = []
        kpt_list = []
        mask_list = []
        arcfaceinp_list = []
        if self.K != 1:
            random_ind = np.random.permutation(16)[:self.K]

            for i in random_ind:
                name = self.data_lines[idx, i]
                image_path = os.path.join(self.imagefolder, name + '.jpg')
                seg_path = os.path.join(self.segfolder, name + '.npy')
                kpt_path = os.path.join(self.kptfolder, name + '.npy')
                arcfaceinp_path = os.path.join(self.arcfacefolder, name + '.npy')

                image = imread(image_path) / 255.
                kpt = np.load(kpt_path)[:, :2]
                mask = self.load_mask(seg_path, image.shape[0], image.shape[1])
                arcfaceinp = np.load(arcfaceinp_path)

                ### crop information
                tform = self.crop(image, kpt)
                ## crop
                cropped_image = warp(image, tform.inverse, output_shape=(self.image_size, self.image_size))
                cropped_mask = warp(mask, tform.inverse, output_shape=(self.image_size, self.image_size))
                cropped_kpt = np.dot(tform.params,
                                     np.hstack([kpt, np.ones([kpt.shape[0], 1])]).T).T  # np.linalg.inv(tform.params)

                # normalized kpt
                cropped_kpt[:, :2] = cropped_kpt[:, :2] / self.image_size * 2 - 1

                images_list.append(cropped_image.transpose(2, 0, 1))
                kpt_list.append(cropped_kpt)
                mask_list.append(cropped_mask)
                arcfaceinp_list.append(arcfaceinp)
        else:
            name = self.data_lines[idx][0]
            image_path = os.path.join(self.imagefolder, name + '.jpg')
            seg_path = os.path.join(self.segfolder, name + '.npy')
            kpt_path = os.path.join(self.kptfolder, name + '.npy')
            arcfaceinp_path = os.path.join(self.arcfacefolder, name + '.npy')

            image = imread(image_path) / 255.
            kpt = np.load(kpt_path)[:, :2]
            mask = self.load_mask(seg_path, image.shape[0], image.shape[1])
            arcfaceinp = np.load(arcfaceinp_path)

            ### crop information
            tform = self.crop(image, kpt)
            ## crop
            cropped_image = warp(image, tform.inverse, output_shape=(self.image_size, self.image_size))
            cropped_mask = warp(mask, tform.inverse, output_shape=(self.image_size, self.image_size))
            cropped_kpt = np.dot(tform.params,
                                 np.hstack([kpt, np.ones([kpt.shape[0], 1])]).T).T  # np.linalg.inv(tform.params)

            # normalized kpt
            cropped_kpt[:, :2] = cropped_kpt[:, :2] / self.image_size * 2 - 1

            images_list.append(cropped_image.transpose(2, 0, 1))
            kpt_list.append(cropped_kpt)
            mask_list.append(cropped_mask)
            arcfaceinp_list.append(arcfaceinp)

        ###
        images_array = torch.from_numpy(np.array(images_list)).type(dtype=torch.float32)  # K,224,224,3
        kpt_array = torch.from_numpy(np.array(kpt_list)).type(dtype=torch.float32)  # K,224,224,3
        mask_array = torch.from_numpy(np.array(mask_list)).type(dtype=torch.float32)  # K,224,224,3
        arcfaceinp_array = torch.from_numpy(np.array(arcfaceinp_list)).type(dtype=torch.float32)  # K,112,112,3
        
        if self.isSingle:
            images_array = images_array.squeeze()
            kpt_array = kpt_array.squeeze()
            mask_array = mask_array.squeeze()
            arcfaceinp_array = arcfaceinp_array.squeeze()
        
        data_dict = {
            'image': images_array,
            'landmark': kpt_array,
            'mask': mask_array,
            'arcface': arcfaceinp_array
        }

        return data_dict

    def crop(self, image, kpt):
        left = np.min(kpt[:, 0]);
        right = np.max(kpt[:, 0]);
        top = np.min(kpt[:, 1]);
        bottom = np.max(kpt[:, 1])

        h, w, _ = image.shape
        old_size = (right - left + bottom - top) / 2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])  # + old_size*0.1])
        # translate center
        trans_scale = (np.random.rand(2) * 2 - 1) * self.trans_scale
        center = center + trans_scale * old_size  # 0.5

        scale = np.random.rand() * (self.scale[1] - self.scale[0]) + self.scale[0]
        size = int(old_size * scale)

        # crop image
        src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                            [center[0] + size / 2, center[1] - size / 2]])
        DST_PTS = np.array([[0, 0], [0, self.image_size - 1], [self.image_size - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)

        # cropped_image = warp(image, tform.inverse, output_shape=(self.image_size, self.image_size))
        # # change kpt accordingly
        # cropped_kpt = np.dot(tform.params, np.hstack([kpt, np.ones([kpt.shape[0],1])]).T).T # np.linalg.inv(tform.params)
        return tform

    def load_mask(self, maskpath, h, w):
        # print(maskpath)
        if os.path.isfile(maskpath):
            mask = np.load(maskpath)
            atts = ['background', 'skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear',
                    'ear_r',
                    'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
            category_matrix = mask.astype(int)
            mask = np.vectorize(self.mask_weight_dict.get)(atts)[category_matrix]
        else:
            print(f"{maskpath} not found!")
            mask = np.ones((h, w))
        return mask

class SmallFaceScape_TrainDataset(Dataset):
    def __init__(self, K, train_path):

        self.K = K # K<16
        self.train_path = train_path
        datafile = os.path.join(train_path, f'train_list_k{K}.npy')
        self.data_lines = np.load(datafile).astype('str')

    def __len__(self):
        return len(self.data_lines)

    def __getitem__(self, idx):
        img_list = [] # 224*224 图片
        skin_mask_list = [] # 皮肤掩膜
        parse_mask_list = [] # 面部解析掩膜
        lmk_list = [] # 人脸关键点（68个）
        M_list = [] # 暂不知道作用

        if self.K != 1:
            random_ind = np.random.permutation(16)[:self.K]
            for i in random_ind:
                name = self.data_lines[idx, i]
                input_data = torch.load(os.path.join(self.train_path, name + '.pt'))
                if 'trans_params' in input_data:
                    input_data.pop('trans_params')
                input_data = {k: v.squeeze() for (k, v) in input_data.items()}

                img_list.append(input_data['img'])
                skin_mask_list.append(input_data['skin_mask'])
                parse_mask_list.append(input_data['parse_mask'])
                lmk_list.append(input_data['lm'])
                M_list.append(input_data['M'])
        else:
            name = self.data_lines[idx][0]
            input_data = torch.load(os.path.join(self.train_path, name + '.pt'))
            if 'trans_params' in input_data:
                input_data.pop('trans_params')
            input_data = {k: v.squeeze() for (k, v) in input_data.items()}

            img_list.append(input_data['img'])
            skin_mask_list.append(input_data['skin_mask'])
            parse_mask_list.append(input_data['parse_mask'])
            lmk_list.append(input_data['lm'])
            M_list.append(input_data['M'])

        # img_array = torch.from_numpy(np.array(img_list)).type(dtype=torch.float32)  # K,224,224,3
            
        img_array = torch.stack(img_list, dim=0)  # K, 224, 224, 3
        skin_mask_array = torch.stack(skin_mask_list, dim=0)
        parse_mask_array = torch.stack(parse_mask_list, dim=0)
        lmk_array = torch.stack(lmk_list, dim=0)
        M_array = torch.stack(M_list, dim=0)

        # img_array = np.array(img_list)
        # skin_mask_array = np.array(skin_mask_list)
        # parse_mask_array = np.array(parse_mask_list)
        # lmk_array = np.array(lmk_list)
        # M_array = np.array(M_list)

        if self.K==1:
            img_array = img_array.squeeze(0)
            skin_mask_array = skin_mask_array.squeeze(0)
            parse_mask_array = parse_mask_array.squeeze(0)
            lmk_array = lmk_array.squeeze(0)
            M_array = M_array.squeeze(0)
        
        data_dict = {
            'img': img_array,
            'skin_mask': skin_mask_array,
            'parse_mask': parse_mask_array,
            'lmk': lmk_array,
            'M': M_array
        }

        return data_dict