import numpy as np
import math
import json
import h5py
import random
from sklearn.model_selection import train_test_split
import yaml
from scipy.io import savemat
import scipy

from feature import Feature_deal


# 预训练：取出所有的训练数据X_train_10Class  在分训练+验证  return X_train, Y_train
# 微调： 取出训练数据X_train_10Class   取测试数据      return X_train, X_test, Y_train, Y_test
def pre_dataset_add_feature():
    # 去读数据
    mat_1 = scipy.io.loadmat(r"feature_dataset/feature_train_90Class.mat")
    X_train = mat_1["f_data"]
    Y_train = np.squeeze(mat_1["label"])

    # 这里的数据经过以下处理：rawIQ->提取人工特征 ,提取人工特征 + 归一化 ->raw IQ归一化-># 拼接人工特征
    # 1.提取人工特征  归一化  删除全是0的特征
    sample_len = X_train.shape[0]
    BPSK_feature = np.array([[0 for _ in range(28)] for _ in range(sample_len)],dtype=np.float64)
    for j in range(sample_len):
        BPSK_feature[j, :] = X_train[j, 4800:, 0]
    mean_axis1 = np.mean(BPSK_feature, axis=0, keepdims=True)
    std_axis1 = np.std(BPSK_feature, axis=0, keepdims=True)
    BPSK_feature = (BPSK_feature - mean_axis1) / (std_axis1 + 1e-8)

    indices_to_keep = [15]
    indices_to_keep = [i for i in range(28) if i not in indices_to_keep]
    BPSK_feature = BPSK_feature[:, indices_to_keep]

    # 2.raw IQ归一化
    sample_len = len(X_train)
    d_x = []
    for i in range(sample_len):
        x_ = X_train[i, :4800]
        max_value = x_.max()
        min_value = x_.min()
        x_ = (x_ - min_value) / (max_value - min_value)
        d_x.append(x_)
    d_x = np.array(d_x)

    # 拼接人工特征
    X = np.zeros((sample_len, 4827, 2), dtype=d_x.dtype)
    for j in range(sample_len):
        z1 = np.concatenate((d_x[j, :, 0], BPSK_feature[j, :]), axis=0)
        z2 = np.concatenate((d_x[j, :, 1], BPSK_feature[j, :]), axis=0)
        X[j, :, 0] = z1
        X[j, :, 1] = z2
    X_train = X.transpose(0, 2, 1)

    train_index_shot = []
    for i in range(90):
        index_classi = [index for index, value in enumerate(Y_train) if value == i]
        train_index_shot += index_classi[0:100]

    X_train = X_train[train_index_shot]
    Y_train = Y_train[train_index_shot].astype(np.uint8)

    return X_train, Y_train

def PreTrainDataset_prepared():
    x = np.load(
        f"../Dataset_4800/X_train_90Class.npy"
    )
    y = np.load(
        f"../Dataset_4800/Y_train_90Class.npy"
    )
    x = x.transpose(0, 2, 1)
    train_index_shot = []
    for i in range(90):
        index_classi = [index for index, value in enumerate(y) if value == i]
        train_index_shot += index_classi[0:100]

    X_train = x[train_index_shot]
    Y_train = y[train_index_shot].astype(np.uint8)
    max_value = X_train.max()
    min_value = X_train.min()

    X_train = (X_train - min_value) / (max_value - min_value)
    return X_train, Y_train


def finetune_dataset_add_feature():
    config = yaml.load(open("config/config.yaml", "r"), Loader=yaml.FullLoader)
    params = config["finetune"]
    k = params["k_shot"]

    # 取数据  # 这里的数据经过以下处理：rawIQ->提取人工特征 ,提取人工特征 + 归一化 ->raw IQ归一化-># 拼接人工特征
    mat_data = scipy.io.loadmat(r"feature_dataset/feature_train_10Class.mat")
    x = mat_data["f_data"]
    y = np.squeeze(mat_data["label"])

    mat_data_test = scipy.io.loadmat(r"feature_dataset/feature_test_10Class.mat")
    X_test = mat_data_test["f_data"]
    Y_test = np.squeeze(mat_data_test["label"])

    # 训练集
    # 1.提取人工特征  归一化  删除全是0的特征
    sample_len = x.shape[0]
    BPSK_feature = np.array([[0 for _ in range(28)] for _ in range(sample_len)],dtype=np.float64)
    for j in range(sample_len):
        BPSK_feature[j, :] = x[j, -28:, 0]
    mean_axis1 = np.mean(BPSK_feature, axis=0, keepdims=True)
    std_axis1 = np.std(BPSK_feature, axis=0, keepdims=True)
    BPSK_feature = (BPSK_feature - mean_axis1) / (std_axis1 + 1e-8)

    indices_to_keep = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 17, 21, 23,24, 25, 26]
    indices_to_keep = [i for i in range(28) if i in indices_to_keep]
    BPSK_feature = BPSK_feature[:, indices_to_keep]

    # 2.raw IQ归一化
    sample_len = len(x)
    d_x = []
    for i in range(sample_len):
        x_ = x[i, :4800]
        max_value = x_.max()
        min_value = x_.min()
        x_ = (x_ - min_value) / (max_value - min_value)
        d_x.append(x_)
    d_x = np.array(d_x)

    # 拼接人工特征
    X = np.zeros((sample_len, 4820, 2), dtype=d_x.dtype)
    for j in range(sample_len):
        z1 = np.concatenate((d_x[j, :, 0], BPSK_feature[j, :]), axis=0)
        z2 = np.concatenate((d_x[j, :, 1], BPSK_feature[j, :]), axis=0)
        X[j, :, 0] = z1
        X[j, :, 1] = z2
    x = X.transpose(0, 2, 1)
    
    x, X_val, y, Y_val = train_test_split(x, y, test_size=0.2, random_state=30)

    # k shot
    finetune_index_shot = []
    for i in range(10):
        index_classi = [index for index, value in enumerate(y) if value == i]
        finetune_index_shot += random.sample(index_classi, k)
    X_train = x[finetune_index_shot]
    Y_train = y[finetune_index_shot]
    
    
    # 1.提取人工特征  归一化  删除全是0的特征
    sample_len = X_test.shape[0]
    BPSK_feature = np.array([[0 for _ in range(28)] for _ in range(sample_len)],dtype=np.float64)
    for j in range(sample_len):
        BPSK_feature[j, :] = X_test[j, -28:, 0]
    mean_axis1 = np.mean(BPSK_feature, axis=0, keepdims=True)
    std_axis1 = np.std(BPSK_feature, axis=0, keepdims=True)
    BPSK_feature = (BPSK_feature - mean_axis1) / (std_axis1 + 1e-8)
    indices_to_keep = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 17, 21, 23,24, 25, 26]
    indices_to_keep = [i for i in range(28) if i in indices_to_keep]
    BPSK_feature = BPSK_feature[:, indices_to_keep]

    # 2.raw IQ归一化
    sample_len = len(X_test)
    d_x = []
    for i in range(sample_len):
        x_ = X_test[i, :4800]
        max_value = x_.max()
        min_value = x_.min()
        x_ = (x_ - min_value) / (max_value - min_value)
        d_x.append(x_)
    d_x = np.array(d_x)

    # 拼接人工特征
    X = np.zeros((sample_len, 4820, 2), dtype=d_x.dtype)
    for j in range(sample_len):
        z1 = np.concatenate((d_x[j, :, 0], BPSK_feature[j, :]), axis=0)
        z2 = np.concatenate((d_x[j, :, 1], BPSK_feature[j, :]), axis=0)
        X[j, :, 0] = z1
        X[j, :, 1] = z2
    X_test = X.transpose(0, 2, 1)
    

    return  X_train, X_test, X_val, Y_train, Y_test, Y_val


def FineTuneDataset_prepared():
    config = yaml.load(open("config/config.yaml", "r"), Loader=yaml.FullLoader)
    params = config["finetune"]
    k = params["k_shot"]
    x = np.load(
       r"E:\Dataset_4800/X_train_10Class.npy"
    )
    y = np.load(
       r"E:\Dataset_4800/Y_train_10Class.npy"
    ) 
    X_test = np.load(
       r"E:\Dataset_4800/X_test_10Class.npy"
    )
    Y_test = np.load(
       r"E:\Dataset_4800/Y_test_10Class.npy"
    ) 
     # 2.raw IQ归一化
    sample_len = len(x)
    d_x = []
    for i in range(sample_len):
        x_ = x[i, :4800]
        max_value = x_.max()
        min_value = x_.min()
        x_ = (x_ - min_value) / (max_value - min_value)
        d_x.append(x_)
    x = np.array(d_x).transpose(0, 2, 1)

    sample_len = len(X_test)
    d_x = []
    for i in range(sample_len):
        x_ = X_test[i, :4800]
        max_value = x_.max()
        min_value = x_.min()
        x_ = (x_ - min_value) / (max_value - min_value)
        d_x.append(x_)
    X_test = np.array(d_x).transpose(0, 2, 1)

    x, X_val, y, Y_val = train_test_split(x, y, test_size=0.1, random_state=30)
     
    
    finetune_index_shot = []
    for i in range(10):
        index_classi = [index for index, value in enumerate(y) if value == i]
        finetune_index_shot += random.sample(index_classi, k)
    X_train = x[finetune_index_shot]
    Y_train = y[finetune_index_shot]

    return X_train, X_test, X_val, Y_train, Y_test, Y_val


# 是训练样本的百分比
def finetune_dataset_add_feature_SSML10CLASS(add_feature =True):
    config = yaml.load(open("config/config.yaml", "r"), Loader=yaml.FullLoader)
    params = config["finetune"]
    # 5%比例 3081*0.01*k = 154
    k = params["k_shot"]

    x = np.load(
       r"E:\Dataset_4800/X_train_10Class.npy"
    )
    y = np.load(
       r"E:\Dataset_4800/Y_train_10Class.npy"
    )

    X_test = np.load(
       r"E:\Dataset_4800/X_test_10Class.npy"
    )
    Y_test = np.load(
       r"E:\Dataset_4800/Y_test_10Class.npy"
    )  

    # 2.raw IQ归一化
    sample_len = len(x)
    d_x = []
    for i in range(sample_len):
        x_ = x[i, :4800]
        max_value = x_.max()
        min_value = x_.min()
        x_ = (x_ - min_value) / (max_value - min_value)
        d_x.append(x_)
    x = np.array(d_x).transpose(0, 2, 1)

    sample_len = len(X_test)
    d_x = []
    for i in range(sample_len):
        x_ = X_test[i, :4800]
        max_value = x_.max()
        min_value = x_.min()
        x_ = (x_ - min_value) / (max_value - min_value)
        d_x.append(x_)
    X_test = np.array(d_x).transpose(0, 2, 1)

    x, X_val, y, Y_val = train_test_split(x, y, test_size=0.1, random_state=30)
    # k shot 每类还要k/class
    sample_len = x.shape[0]
    # k = int(k * 0.01 * sample_len *0.1)
    finetune_index_shot = []
    for i in range(10):
        index_classi = [index for index, value in enumerate(y) if value == i]
        finetune_index_shot += random.sample(index_classi, k)
    X_train = x[finetune_index_shot]
    Y_train = y[finetune_index_shot]

    return X_train, X_test, X_val, Y_train, Y_test, Y_val

from sklearn.preprocessing import StandardScaler
if __name__ == "__main__":
    """
    # 去读数据
    x = np.load(
       r"E:\Dataset_4800/X_test_10Class.npy"
    )
    y = np.load(
       r"E:\Dataset_4800/Y_test_10Class.npy"
    )

    add_feature = True
    train_index_shot = []
    for i in range(10): 
        index_classi = [index for index, value in enumerate(y) if value == i]
        train_index_shot += index_classi
    # (3081,4800,2)
    X_train = x[train_index_shot]
    Y_train = y[train_index_shot].astype(np.uint8)
    Y_train = Y_train.reshape(-1)

    if add_feature == True:
        # 提取人工特征  归一化  删除全是0的特征
        # 提取人工特征 + 归一化 + 删除全是0的特征
        sample_len = X_train.shape[0]
        BPSK_feature = np.array([[0 for _ in range(26)] for _ in range(sample_len)])
        for j in range(sample_len):
            BPSK_feature[j, :] = Feature_deal(X_train[j, :, 0])
        mean_axis1 = np.mean(BPSK_feature, axis=0, keepdims=True)
        std_axis1 = np.std(BPSK_feature, axis=0, keepdims=True)
        BPSK_feature = (BPSK_feature - mean_axis1) / (std_axis1 + 1e-8)
        indices_to_keep = [
            0,
            2,
            4,
            6,
            10,
            11,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
        ]
        indices_to_keep = [i for i in range(26) if i not in indices_to_keep]
        BPSK_feature = BPSK_feature[:, indices_to_keep]

    # raw IQ归一化
    sample_len = len(X_train)
    d_x = []
    for i in range(sample_len):
        x_ = X_train[i, :]
        max_value = x_.max()
        min_value = x_.min()
        x_ = (x_ - min_value) / (max_value - min_value)
        d_x.append(x_)
    d_x = np.array(d_x)

    if add_feature == True:
        # 拼接人工特征
        X = np.zeros((sample_len, 4807, 2), dtype=d_x.dtype)
        for j in range(sample_len):
            z1 = np.concatenate((d_x[j, :, 0], BPSK_feature[j, :]), axis=0)
            z2 = np.concatenate((d_x[j, :, 1], BPSK_feature[j, :]), axis=0)
            X[j, :, 0] = z1
            X[j, :, 1] = z2
        X = X.transpose(0, 2, 1)

        # (9000, 2 , 4807)
        savemat(
            r"./feature_dataset/feature_test_10Class.mat",
            {"data": X, "label": Y_train},
        )
"""

    mat_1 = scipy.io.loadmat(r"feature_dataset/feature_train_90Class.mat")
    # 这里的数据经过以下处理：rawIQ->提取人工特征 ,提取人工特征 + 归一化 ->raw IQ归一化-># 拼接人工特征
    X_f = mat_1["f_data"]
    Y_f = np.squeeze(mat_1["label"])
    # 将第二维的数据 reshape 成 (n_samples, n_features)
    X_reshaped = X_f.reshape(X_f.shape[0], -1)

    # 创建 StandardScaler 对象
    scaler = StandardScaler()

    # 对第二维的数据进行标准化处理
    X_processed = scaler.fit_transform(X_reshaped)

    # 将处理后的数据重新 reshape 成原来的形状
    X_processed = X_processed.reshape(X_f.shape)

    # 确认处理后的数据形状
    print("处理后的数据形状:", X_processed.shape)
