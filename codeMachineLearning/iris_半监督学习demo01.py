from sklearn.datasets import load_iris
from sklearn.semi_supervised import LabelPropagation
from sklearn.model_selection import train_test_split
import numpy as np

# 加载数据集
iris = load_iris()
X = iris.data
y = iris.target

# 人为将部分标签设置为-1（表示未标记）
rng = np.random.RandomState(42)
mask_unlabeled = rng.rand(y.shape[0]) < 0.5  # 随机选择50%的数据为未标记
y[mask_unlabeled] = -1  # 未标记数据的标签设为-1

# 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建并训练LabelPropagation模型
label_prop_model = LabelPropagation()
label_prop_model.fit(X_train, y_train)

# 预测测试集
y_pred = label_prop_model.predict(X_test)

# 计算准确率
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test[y_test != -1], y_pred[y_test != -1])
print(f'Accuracy: {accuracy * 100:.2f}%')
