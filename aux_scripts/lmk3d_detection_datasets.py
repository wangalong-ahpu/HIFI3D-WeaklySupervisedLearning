import os
import os
import time
import argparse
import torch
import numpy as np

from ..RGB_Fitting.dataset.fit_dataset import FitDataset
from ..RGB_Fitting.utils.data_utils import tensor2np, img3channel, draw_mask, draw_landmarks, save_img

if __name__ == '__main__':
    # datasets_path_list = ["/media/wang/SSD_2/Datasets/vggface2_new"]
    datasets_path_list = ["/media/wang/SSD_2/Datasets/FaceScape/small_facescape_new"]
    #"/media/wang/SSD_2/Datasets/FaceScape/fmview_train_new", "/media/wang/SSD_2/Datasets/FaceScape/small_facescape_new","/media/wang/SSD_2/Datasets/vggface2_new",

    # 制作lmk for hifi3d
    for dataset_path in datasets_path_list:
        # 图片文件夹
        imgs_folder = os.path.join(dataset_path, 'image')
        # 关键点输出文件夹
        lmk68_out_folder = os.path.join(dataset_path, 'lmk68')
        os.makedirs(lmk68_out_folder, exist_ok=True)
        
        checkpoints_dir = "../checkpoints"
        topo_dir = "./topo_assets"


        # 



        dataset_op = FitDataset(lm_detector_path=os.path.join(checkpoints_dir, 'lm_model/68lm_detector.pb'),
                            mtcnn_detector_path=os.path.join(checkpoints_dir, 'mtcnn_model/mtcnn_model.pb'),
                            parsing_model_pth=os.path.join(checkpoints_dir, 'parsing_model/79999_iter.pth'),
                            parsing_resnet18_path=os.path.join(checkpoints_dir,
                                                               'resnet_model/resnet18-5c106cde.pth'),
                            lm68_3d_path=os.path.join(topo_dir, 'similarity_Lm3D_all.mat'),
                            batch_size=1,
                            device='cuda')

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.output_dir + '_vis', exist_ok=True)
    fnames = [
        fn for fn in sorted(os.listdir(args.input_dir))
        if fn.endswith('.jpg') or fn.endswith('.png') or fn.endswith('.jpeg')
    ]
    for fn in fnames:
        tic = time.time()
        basename = fn[:fn.rfind('.')]
        input_data = dataset_op.get_input_data(os.path.join(args.input_dir, fn))
        if input_data is None:
            continue

        torch.save(input_data, os.path.join(args.output_dir, f'{basename}.pt'))

        input_img = tensor2np(input_data['img'][:1, :, :, :])
        skin_img = tensor2np(input_data['skin_mask'][:1, :, :, :])
        skin_img = img3channel(skin_img)
        parse_mask = tensor2np(input_data['parse_mask'][:1, :, :, :], dst_range=1.0)
        parse_img = draw_mask(input_img, parse_mask)
        gt_lm = input_data['lm'][0, :, :].detach().cpu().numpy()
        gt_lm[..., 1] = input_img.shape[0] - 1 - gt_lm[..., 1]
        lm_img = draw_landmarks(input_img, gt_lm, color='b')
        combine_img = np.concatenate([input_img, skin_img, parse_img, lm_img], axis=1)
        save_img(combine_img, os.path.join(args.output_dir + '_vis', f'{basename}.png'))

        toc = time.time()
        print(f'Process image: {fn} done, took {toc - tic:.4f} seconds.')