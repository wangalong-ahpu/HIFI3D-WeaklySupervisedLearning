import os
import torch
import numpy as np
from queue import Queue
import threading
import shutil
from dataset.fit_dataset import FitDataset
from utils.data_utils import tensor2np, img3channel, draw_mask, draw_landmarks, save_img

class task:
    def __init__(self,input_dir,output_dir,remove_dir,save_img):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.remove_dir = remove_dir
        self.save_img = save_img

# 全局变量，用于通知线程退出
exit_flag = False
def task_worker(q):
    checkpoints_dir = "../checkpoints"
    topo_dir = "../topo_assets"
    dataset_op = FitDataset(lm_detector_path=os.path.join(checkpoints_dir, 'lm_model/68lm_detector.pb'),
                        mtcnn_detector_path=os.path.join(checkpoints_dir, 'mtcnn_model/mtcnn_model.pb'),
                        parsing_model_pth=os.path.join(checkpoints_dir, 'parsing_model/79999_iter.pth'),
                        parsing_resnet18_path=os.path.join(checkpoints_dir,
                                                            'resnet_model/resnet18-5c106cde.pth'),
                        lm68_3d_path=os.path.join(topo_dir, 'similarity_Lm3D_all.mat'),
                        batch_size=1,
                        device='cpu')
    thread_id = threading.get_ident()
    while not exit_flag:
        task = q.get()
        if task is None:
            q.task_done()
            continue
        try:
            os.makedirs(task.output_dir, exist_ok=True)
            os.makedirs(task.remove_dir, exist_ok=True)
            fnames = [
                fn for fn in sorted(os.listdir(task.input_dir))
                if fn.endswith('.jpg') or fn.endswith('.png') or fn.endswith('.jpeg') or fn.endswith('.bmp')
            ]
            
            for fn in fnames:
                basename = fn[:fn.rfind('.')]
                if os.path.exists(f"{os.path.join(task.output_dir, f'{basename}.pt')}"): # 存在则跳过
                    continue
                try:
                    # print(os.path.join(task.input_dir, fn))
                    input_data = dataset_op.get_input_data(os.path.join(task.input_dir, fn))
                    if input_data==None: # 未检测到人脸则跳过
                        shutil.move(os.path.join(task.input_dir, fn),task.remove_dir)
                        continue
                    torch.save(input_data, os.path.join(task.output_dir, f'{basename}.pt'))
                    if task.save_img:# 保存结果
                        input_img = tensor2np(input_data['img'][:1, :, :, :])
                        skin_img = tensor2np(input_data['skin_mask'][:1, :, :, :])
                        skin_img = img3channel(skin_img)
                        parse_mask = tensor2np(input_data['parse_mask'][:1, :, :, :], dst_range=1.0)
                        parse_img = draw_mask(input_img, parse_mask)
                        gt_lm = input_data['lm'][0, :, :].detach().cpu().numpy()
                        gt_lm[..., 1] = input_img.shape[0] - 1 - gt_lm[..., 1]
                        lm_img = draw_landmarks(input_img, gt_lm, color='b')
                        combine_img = np.concatenate([input_img, skin_img, parse_img, lm_img], axis=1)
                        save_img(combine_img, os.path.join(task.output_dir, f'{basename}.png'))
                except:
                    shutil.move(os.path.join(task.input_dir, fn),task.remove_dir)
                    continue
            print(f"{thread_id}\t{task.input_dir}\t all done")
            q.task_done()
            continue
        except:
            print(f"{thread_id}\t{task.input_dir}\t error")
            q.task_done()
            continue

if __name__ == '__main__':
    datasets_path_list = ["/root/autodl-tmp/datasets/small_facescape_new"]
    #"/media/wang/SSD_2/Datasets/FaceScape/fmview_train_new", "/media/wang/SSD_2/Datasets/FaceScape/small_facescape_new","/media/wang/SSD_2/Datasets/vggface2_new",

    # 制作lmk for hifi3d
    task_queue = Queue()
    # threads = []

    # 创建多线程
    for i in range(18):
        t = threading.Thread(target=task_worker, args=(task_queue,))
        # threads.append(t)
        t.start()

    for dataset_path in datasets_path_list:
        # 图片文件夹
        imgs_folder = os.path.join(dataset_path, 'image')
        # 关键点输出文件夹
        lmk68_out_folder = os.path.join(dataset_path, 'lmk68'); os.makedirs(lmk68_out_folder, exist_ok=True)
        img_remove_folder = os.path.join(dataset_path, 'remove'); os.makedirs(img_remove_folder, exist_ok=True)
        for sub_imgs_folder in os.listdir(imgs_folder):
            input_dir = os.path.join(imgs_folder, sub_imgs_folder)
            output_dir = os.path.join(lmk68_out_folder, sub_imgs_folder)
            remove_dir = os.path.join(img_remove_folder, sub_imgs_folder)
            task_t = task(input_dir=input_dir,output_dir=output_dir,remove_dir=remove_dir,save_img=True)
            task_queue.put(task_t)
    
    # 等待所有任务完成
    task_queue.join()
    # 通知线程退出
    exit_flag = True
    print("结束")