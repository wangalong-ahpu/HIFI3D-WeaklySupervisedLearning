unset LD_LIBRARY_PATH # 解决报错：CUBLAS_STATUS_INVALID_VALUE when calling `cublasSgemmStridedBatched( handle, opa, opb, m\
# 第一次运行时需要执行
python demos/demo_reconstruct.py -i ../TestSamples/exp -s ../TestSamples/exp/results
python demos/demo_reconstruct.py -i /media/wang/SSD_2/demo/TestSamples/REALY_TEST -s /media/wang/SSD_2/demo/TestSamples/REALY_TEST/results
python demos/demo_reconstruct.py -i /media/wang/SSD_2/demo/TestSamples/mp4/1.mp4 -s /media/wang/SSD_2/demo/TestSamples/mp4/results
python demos/demo_transfer.py --rasterizer_type pytorch3d
python demos/demo_teaser.py --ra sterizer_type pytorch3d


# 训练
python main_train.py --cfg configs/release_version/deca_pretrain.yml
python main_train.py --cfg configs/release_version/deca_coarse.yml
python main_train.py --cfg configs/wang/deca_pretrain_convnext_v2_191.yml

#4090
python main_train.py --cfg configs/4090/coarse.yml
python main_train.py --cfg configs/4090/pretrain.yml
#1080
python main_train.py --cfg configs/1080/deca_pretrain_convnext_v2_235.yml
python main_train.py --cfg configs/1080/coarse.yml
python main_train.py --cfg configs/1080/pretrain.yml
python main_train.py --cfg configs/1080/detail.yml

#auto
python main_train.py --cfg configs/auto/deca_pretrain_convnext_v2_235.yml
python main_train.py --cfg configs/auto/coarse.yml
python main_train.py --cfg configs/auto/pretrain.yml
python main_train.py --cfg configs/auto/detail.yml

# 可视化结果
tensorboard --logdir /media/c723/SSD_1/demo/DECA_TR/coarse//logs
tensorboard --logdir /media/c723/SSD_1/demo/DECA_TR/pretrain/4090_resnet50_16_235_0118/logs
tensorboard --logdir /media/c723/SSD_1/demo/DECA/train/step_2_coarse/logs --port 6006

fuser -v /dev/nvidia* |awk '{for(i=1;i<=NF;i++)print "kill -9 " $i;}' | sh
export CUDA_HOME=/usr/local/cuda
unset LD_LIBRARY_PATH

source /etc/network_turbo
ipconfig /flushdns(windows)
# 官方教程
git init
git add README.md

git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/wangalong-ahpu/AlbedoGAN.git
git push -u origin main
