from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# 加载鸢尾花数据集
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 将数据集拆分为训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40) # random_state是随机种子，保证每次划分的数据集都是相同的
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# # 创建逻辑回归模型，会根据数据决定使用多分类还是二分类
# 其实多分类情况下是自动去用softmax回归，sklearn包并没有专门去封装softmax回归
# 当然也可以用逻辑回归OVR去做多分类，可以通过超参数控制
# lr = LogisticRegression(multi_class='multinomial') # softmax回归做多分类
# lr = LogisticRegression(multi_class='ovr')# OVR做多分类，也就是转成多个二分类
# lr = LogisticRegression(max_iter=1000)# 最大迭代次数

lr = LogisticRegression()

# # 训练模型
lr.fit(X_train, y_train)

# # 预测测试集
y_pred = lr.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
# 准确率
print(f'Accuracy: {accuracy * 100:.2f}%')