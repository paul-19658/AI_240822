# 导入必要的库
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import datasets
# 加载数据集
iris = datasets.load_iris()
X = iris.data
y = iris.target
# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# 创建逻辑回归模型
logreg = LogisticRegression()
# 训练模型
logreg.fit(X_train, y_train)
# 使用训练好的模型进行预测
y_pred = logreg.predict(X_test)
# 计算并打印准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
