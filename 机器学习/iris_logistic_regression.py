from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# 加载鸢尾花数据集
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 将数据集拆分为训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# # 创建逻辑回归模型
lr = LogisticRegression()

# # 训练模型
lr.fit(X_train, y_train)

# # 预测测试集
y_pred = lr.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
# 准确率
print(f'Accuracy: {accuracy * 100:.2f}%')