import numpy as np
import matplotlib.pyplot as plt

from feature import Feature_deal

# 加载数据
x = np.load(f"E:\yiiiiiiiiii\yan\组会\paper\Dataset\Dataset_4800/X_train_90Class.npy")
y = np.load(f"E:\yiiiiiiiiii\yan\组会\paper\Dataset\Dataset_4800/Y_train_90Class.npy")

# 选择前10个类别
num_classes = 10
num_samples_per_class = 15
train_index_shot = []
for i in range(num_classes):
    index_classi = [index for index, value in enumerate(y) if value == i]
    train_index_shot += index_classi[0:num_samples_per_class]

# 提取特征并绘图

fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(15, 20))
for class_index in range(num_classes):
    # 获取当前类别的信号数据
    class_data_indices = [i for i, label in enumerate(y[train_index_shot]) if label == class_index]
    class_data = x[train_index_shot][class_data_indices]

    # 提取特征 (15,26)
    class_features = np.array([Feature_deal(signal[:, 0]) for signal in class_data])
    ax = axes[class_index // 2, class_index % 2]
    # 绘制特征图
    for sample_index in range(class_features.shape[0]):
        ax.plot(range(class_features.shape[1]), class_features[sample_index, :],
                label=f"Class {class_index}, Sample {sample_index}")

ax.set_xlabel('Sample Index')
ax.set_ylabel('Feature Value')
ax.set_title('Signal Features')
ax.legend()
plt.show()