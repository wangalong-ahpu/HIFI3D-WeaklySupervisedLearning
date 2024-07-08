from tqdm import tqdm, trange
import os
import numpy as np

def extract_single_filenames(folder_path, output_file):
    file_list = []
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    for sub_dir in subfolders:
        fnames = [fn for fn in sorted(os.listdir(os.path.join(folder_path, sub_dir))) if fn.endswith('.pt')]
        current_folder_files = []
        for filename in fnames:
            file_path = os.path.join(os.path.basename(sub_dir), filename)
            current_folder_files.append(file_path)
        file_list.extend(current_folder_files)
    # 去除文件名的后缀
    file_list = [os.path.splitext(file_path)[0] for file_path in file_list]
    # 保存为.npy文件
    np.save(output_file, np.array(file_list).reshape(-1,1))


def extract_K_filenames(folder_path, output_file, K):
    file_list = []
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    for sub_dir in subfolders:
        fnames = [fn for fn in sorted(os.listdir(os.path.join(folder_path, sub_dir))) if fn.endswith('.pt')]
        current_folder_files = []
        for filename in fnames:
            file_path = os.path.join(os.path.basename(sub_dir), filename)
            current_folder_files.append(file_path)
            if len(current_folder_files) == K:
                break
        if len(current_folder_files) == K:
            file_list.extend(current_folder_files)
        else:
            continue
    # 去除文件名的后缀
    file_list = [os.path.splitext(file_path)[0] for file_path in file_list]
    # 保存为.npy文件
    np.save(output_file, np.array(file_list).reshape(-1, K))


if __name__ == '__main__':
    datasets_path_list = ["/media/wang/SSD_1/Datasets/sm_fs"]
    #"/media/wang/SSD_2/Datasets/FaceScape/fmview_train_new", "/media/wang/SSD_2/Datasets/FaceScape/small_facescape_new","/media/wang/SSD_2/Datasets/vggface2_new",
    
    K = 2 # 抽取一个个体的图片数
    for dataset_folder in datasets_path_list:
        output_file = os.path.join(dataset_folder, f'train_list_k{K}.npy')
        extract_single_filenames(dataset_folder, output_file) if K==1 else extract_K_filenames(dataset_folder, output_file, K)
 
    # 验证
    x = np.load(f"/media/wang/SSD_1/Datasets/sm_fs/train_list_k{K}.npy")
    print(x.shape)
