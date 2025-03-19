import sys
import numpy as np
import scipy
import torch
from sklearn.manifold import TSNE
from torch.utils.data import TensorDataset, DataLoader
import matplotlib as mpl


mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns

# from get_dataset_10label import *
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sklearn.metrics as sm
from sklearn import manifold

sys.path.insert(0, "models")


def scatter(features, targets, subtitle=None, n_classes=10):
    palette = np.array(sns.color_palette("hls", n_classes))  # "hls",
    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect="equal")
    sc = ax.scatter(
        features[:, 0], features[:, 1], lw=0, s=40, c=palette[targets, :]
    )  #
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis("off")
    ax.axis("tight")

    txts = []
    for i in range(n_classes):
        xtext, ytext = np.median(features[targets == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects(
            [PathEffects.Stroke(linewidth=5, foreground="w"), PathEffects.Normal()]
        )
        txts.append(txt)
    plt.savefig(f"Visualization/{n_classes}classes_{subtitle}.png", dpi=600)


def visualize_data(data, labels, title, num_clusters):  # feature visualization
    labels = labels.astype(int)
    tsne = manifold.TSNE(n_components=2)  # init='pca'
    data_tsne = tsne.fit_transform(data)
    fig = plt.figure()
    plt.scatter(
        data_tsne[:, 0],
        data_tsne[:, 1],
        lw=0,
        s=10,
        c=labels,
        cmap=plt.cm.get_cmap("jet", num_clusters),
    )
    plt.colorbar(ticks=range(num_clusters))
    fig.savefig(title, dpi=600)


def TestDataset_prepared(k, rand_num):

    mat_data_test = scipy.io.loadmat(r"feature_dataset/feature_test_10Class.mat")
    X_test = mat_data_test["f_data"]
    Y_test = np.squeeze(mat_data_test["label"])

    # 1.提取人工特征  归一化  删除全是0的特征
    sample_len = X_test.shape[0]
    BPSK_feature = np.array(
        [[0 for _ in range(28)] for _ in range(sample_len)], dtype=np.float64
    )
    for j in range(sample_len):
        BPSK_feature[j, :] = X_test[j, -28:, 0]
    mean_axis1 = np.mean(BPSK_feature, axis=0, keepdims=True)
    std_axis1 = np.std(BPSK_feature, axis=0, keepdims=True)
    BPSK_feature = (BPSK_feature - mean_axis1) / (std_axis1 + 1e-8)
    indices_to_keep = [
        0,
        1,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        14,
        16,
        17,
        21,
        23,
        24,
        25,
        26,
    ]
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

    return X_test, Y_test


def obtain_embedding_feature_map(model, test_dataloader):
    model.eval()
    device = torch.device("cuda:0")
    with torch.no_grad():
        feature_map = []
        target_output = []
        for data, target in test_dataloader:
            # target = target.long()
            if torch.cuda.is_available():
                data = data.to(device)
                # target = target.to(device)
            output = model(data)
            feature_map[len(feature_map) : len(output[0]) - 1] = output[0].tolist()
            target_output[len(target_output) : len(target) - 1] = target.tolist()
        feature_map = torch.Tensor(feature_map)
        target_output = np.array(target_output)
    return feature_map, target_output


# from models.encoder_and_projection import Encoder_and_projection
# from models.complexcnn import ComplexConv
# import models


def main():
    X_test, Y_test = TestDataset_prepared(30, rand_num=50)
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    model = torch.load(r"model_weight\online_network_PT_0-89_FT_90-99_5shot_32.pth")

    X_test_embedding_feature_map, target = obtain_embedding_feature_map(
        model, test_dataloader
    )

    tsne = TSNE(n_components=2)
    eval_tsne_embeds = tsne.fit_transform(
        torch.Tensor.cpu(X_test_embedding_feature_map)
    )
    scatter(
        eval_tsne_embeds,
        target.astype("int64"),
        "KDM",
        10,
    )
    visualize_data(
        X_test_embedding_feature_map,
        target.astype("int64"),
        "Visualization/KDM",
        10,
    )
    print(
        sm.silhouette_score(
            X_test_embedding_feature_map,
            target,
            sample_size=len(X_test_embedding_feature_map),
            metric="euclidean",
        )
    )


if __name__ == "__main__":
    main()
