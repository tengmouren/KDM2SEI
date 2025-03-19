import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

# 所有人工特征总和
# # 加载已保存的特征数组
# BPSK_feature = np.load('BPSK_feature.npy')
#
# # 定义样本的分割点
# sample_ranges = [0, 249, 409, 517, 940, 1124, 1611, 1909, 2336, 2735, 3080]
#
# # 创建多个子图
# fig, axes = plt.subplots(10, 1, figsize=(15, 50))  # 创建10个子图
#
# # 为每个子图绘制特征曲线
# for i in range(10):
#     ax = axes[i]
#     start_index = sample_ranges[i]
#     # 对于最后一个子图，结束索引是数组的最后一个元素
#     end_index = sample_ranges[i + 1] if i < 9 else len(BPSK_feature)
#
#     for feature_idx in range(BPSK_feature.shape[1]):
#         ax.plot(range(start_index, end_index), BPSK_feature[start_index:end_index, feature_idx],
#                 label=f'Feature {feature_idx + 1}')
#
#     ax.set_title(f'Samples {start_index} to {end_index - 1}')
#     ax.set_xlabel('Sample Index')
#     ax.set_ylabel('Feature Amplitude')
#     ax.legend(loc='upper right')
#
# plt.tight_layout()
# plt.show()
#
# # 将生成的图保存到文件
# plt.savefig('/mnt/data/BPSK_features_visualization.png')




# 特征工程
# 判断使用多少特征最好


# 评估不同特征数量下的分类器性能
# 我们将使用决策树作为分类器，因为它易于实现，并且可以直接从特征选择中受益

# 初始化一些将用于记录的变量

# 重新加载特征数组
BPSK_feature = np.load('BPSK_feature.npy')
labels = np.concatenate([
    np.full((250), 0),      # 第1个图的样本标签是0
    np.full((160), 1),      # 第2个图的样本标签是1
    np.full((108), 2),      # 第3个图的样本标签是2
    np.full((423), 3),      # 第4个图的样本标签是3
    np.full((184), 4),      # 第5个图的样本标签是4
    np.full((487), 5),      # 第6个图的样本标签是5
    np.full((298), 6),      # 第7个图的样本标签是6
    np.full((427), 7),      # 第8个图的样本标签是7
    np.full((399), 8),      # 第9个图的样本标签是8
    np.full((345), 9),      # 第10个图的样本标签是9
])
feature_scores_mlp = {}

# 使用相同的最大特征数
for k in range(1, 27 + 1):
    # 使用SelectKBest选择k个最佳特征
    selector = SelectKBest(score_func=f_classif, k=k)
    X_new = selector.fit_transform(BPSK_feature, labels)

    # 创建MLP分类器，这里只用了一个隐藏层，大小为100，您可以根据需要调整
    clf = MLPClassifier(hidden_layer_sizes=(100,), random_state=1, max_iter=300)

    # 评估当前特征集的分类器准确率
    scores = cross_val_score(clf, X_new, labels, cv=5, scoring='accuracy')

    # 记录平均准确率
    feature_scores_mlp[k] = scores.mean()

feature_scores_mlp_sorted = sorted(feature_scores_mlp.items(), key=lambda item: item[1], reverse=True)
best_num_features_mlp = feature_scores_mlp_sorted[0][0]
best_score_mlp = feature_scores_mlp_sorted[0][1]
# 结果:分数表明，当我们将特征数量从 1 个增加到 20 个时，分类器的准确性会提高。超过 20 个特征后，准确率的提高会趋于稳定，并且添加更多特征不会显着提高性能。
# 准确率为0.6854
(best_num_features_mlp, best_score_mlp, feature_scores_mlp)

# 决策树
"""
# feature_scores = {}
# max_features = min(BPSK_feature.shape[1], 15)  # 将最大特征数限制在15以内
#
# for k in range(1, max_features + 1):
#     # 使用SelectKBest选择k个最佳特征
#     selector = SelectKBest(score_func=f_classif, k=k)
#     X_new = selector.fit_transform(BPSK_feature, labels)
#
#     # 创建决策树分类器
#     clf = DecisionTreeClassifier(random_state=0)
#
#     # 评估当前特征集的分类器准确率
#     scores = cross_val_score(clf, X_new, labels, cv=5, scoring='accuracy')
#
#     # 记录平均准确率
#     feature_scores[k] = scores.mean()
#
# feature_scores_sorted = sorted(feature_scores.items(), key=lambda item: item[1], reverse=True)
# best_num_features = feature_scores_sorted[0][0]
# best_score = feature_scores_sorted[0][1]
#
# (best_num_features, best_score, feature_scores)
# 结果:分数表明，当我们将特征数量从 1 个增加到 13 个时，分类器的准确性会提高。超过 13 个特征后，准确率的提高会趋于稳定，并且添加更多特征不会显着提高性能。
# (13,
#  0.5167147277357975,
#  {1：0.26258340524953167，
#   2：0.3252231156201983，
#   3：0.3709870761329433，
#   4：0.38786124734260874，
#   5：0.4410911616746301，
#   6：0.492695068302848，
#   7：0.5050327306405102，
#   8: 0.5111905112715485,
#   9：0.5079484939695635，
#   10: 0.5115246584857605,
#   11：0.5053474078595634，
#   12: 0.5111941947841461,
#   13: 0.5167147277357975,
#   14: 0.5163863688985245,
#   15: 0.5079453366730513})
"""

# 使用SelectKBest选择20个最佳特征
selector = SelectKBest(score_func=f_classif, k=20)
X_new = selector.fit_transform(BPSK_feature, labels)

# 获取选定的特征索引
selected_features = selector.get_support(indices=True)

# 使用MLP分类器评估选择的20个特征的分类性能
mlp_clf = MLPClassifier(random_state=0, max_iter=300)  # 增加迭代次数以确保收敛

# 使用交叉验证计算平均准确率
scores = cross_val_score(mlp_clf, X_new, labels, cv=5, scoring='accuracy')

# 返回选择的特征和平均准确率 (数组([0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 17, 21, 23,24, 25, 26]),
(selected_features, scores.mean())



