import gzip
import struct
import numpy as np

# 读取 MNIST 图像数据（训练集或测试集）
def read_images(file_path):
    with gzip.open(file_path, 'rb') as f:
        # 读取文件头部的元数据
        magic, num_images, num_rows, num_cols = struct.unpack(">IIII", f.read(16))
        
        # 读取图像数据
        image_data = np.frombuffer(f.read(), dtype=np.uint8)
        
        # 将数据重塑为 (num_images, num_rows, num_cols)
        images = image_data.reshape(num_images, num_rows, num_cols)
        
        return images

# 读取 MNIST 标签数据（训练集或测试集）
def read_labels(file_path):
    with gzip.open(file_path, 'rb') as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

# 图像数据
x_train = read_images('data/train-images-idx3-ubyte.gz').reshape(-1, 28, 28)  # 重塑为 (样本数, 28, 28)
x_val = read_images('data/t10k-images-idx3-ubyte.gz').reshape(-1, 28, 28)
    
# 标签数据
y_train = read_labels('data/train-labels-idx1-ubyte.gz')
y_val = read_labels('data/t10k-labels-idx1-ubyte.gz')

np.savez_compressed('data/mnist.npz', x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)