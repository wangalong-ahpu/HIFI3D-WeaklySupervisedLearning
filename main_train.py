''' 
基于弱监督学习的三维人脸重建
拓扑: HIFI3D++ 
纹理: HIFI3D++ 
关键点: 68、235
'''
import os, sys
import numpy as np
import yaml
import torch
import torch.backends.cudnn as cudnn
import torch
import shutil
from copy import deepcopy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
np.random.seed(0)

def main(cfg):
    # 创建输出目录
    os.makedirs(os.path.join(cfg.output_dir, cfg.train.log_dir), exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, cfg.train.vis_dir), exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, cfg.train.val_vis_dir), exist_ok=True)
    with open(os.path.join(cfg.output_dir, cfg.train.log_dir, 'full_config.yaml'), 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    shutil.copy(cfg.cfg_file, os.path.join(cfg.output_dir, 'config.yaml'))

    # cudnn 设置
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True

    # 准备训练
    # vifc model
    from vifrlib.vifr import VIFR
    from vifrlib.trainer_vifr import Trainer
    cfg.rasterizer_type = 'pytorch3d'
    vifr = VIFR(cfg)
    trainer = Trainer(model=vifr, config=cfg)

    # 开始训练
    trainer.fit()

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

    from vifrlib.utils.config import parse_args
    cfg = parse_args()
    if cfg.cfg_file is not None:
        exp_name = cfg.cfg_file.split('/')[-1].split('.')[0]  # deca_pretrain or deca_coarse
        cfg.exp_name = exp_name
    main(cfg)
# python main_train.py --cfg configs/1080/pretrain.yml