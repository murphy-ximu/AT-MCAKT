import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from time import time
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter


if __name__ == "__main__":
    X = np.array([[0, 0, 0, 1], [0, 1, 1, 2], [1, 0, 1, 0], [1, 1, 1, 1]])

    '''X是特征，不包含target;X_tsne是已经降维之后的特征'''
    t0 = time()
    tsne = TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(X)
    print(X_tsne)
    t1 = time()
    print("t-SNE: %.2g sec" % (t1 - t0))  # 算法用时


    '''嵌入空间可视化'''
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    plt.figure(figsize=(8, 8))

    # tgt = np.loadtxt('target.txt')
    tgt = np.array([0, 0, 1, 3])

    scatter = plt.scatter(X_norm[:, 0], X_norm[:, 1], c=tgt)
    plt.legend(handles=scatter.legend_elements()[0], labels=["neutral", "positive", "negative"], title="classes")
    plt.show()