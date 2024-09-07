from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 加载数据集
iris = load_iris()
X = iris.data

# 创建K-means模型并进行训练
kmeans = KMeans(n_clusters=2, random_state=0)  # 假设我们想分成3个簇
kmeans.fit(X)

# 获取簇标签
labels = kmeans.labels_

# 绘制聚类结果
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-means Clustering')
plt.show()
