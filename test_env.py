# 测试pytorch
import torch
a = torch.cuda.is_available()
print(a)
ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)
print(torch.cuda.get_device_name(0))
print(torch.rand(3, 3).cuda()) 

# 测试tensorflow
# import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
# hello = tf.constant('Hello, TensorFlow!')
# sess = tf.compat.v1.Session()
# print(sess.run(hello))

# 测试pytorch3d
import pytorch3d
print(pytorch3d.__version__)