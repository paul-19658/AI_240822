from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

'''
    做机器学习的一个基本的流程
    1 加载数据
    2 切分训练集测试集
    3 创建算法对象
    4 使用训练集训练
    5 使用测试集进行测试
    6 进行评估
'''


# 加载糖尿病数据集
diabetes = datasets.load_diabetes()
X=diabetes.data # 获取特征数据
y=diabetes.target # 获取目标数据

# print(X)
# print(y)

# 将数据集拆分为训练集和测试集
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

#创建一个多元线性回归算法对象（多元线性回归模型）
lr=LinearRegression()

# 使用训练集训练模型
lr.fit(X_train,y_train)


# 使用测试集进行预测
y_pred_test = lr.predict(X_test)
y_pred_train = lr.predict(X_train)

# 计算模型的均方误差
print('均方误差：%.2f' % mean_squared_error(y_test,y_pred_test))
print('均方误差（训练集：%.2f' % mean_squared_error(y_train,y_pred_train))


# print(lr.score(X_test,y_test))
