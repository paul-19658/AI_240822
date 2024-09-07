from sklearn import datasets  # 此模块包含了很多用于机器学习的数据集。

diabetes = datasets.load_diabetes()  # 使用load_diabetes函数加载糖尿病数据集。

# 该数据集包含442行数据和10个属性值，分别是年龄(Age)、性别(Sex)、体质指数(Body    mass index)、平均血压(Average Blood Pressure)和一年后疾病级数指标(S1~S6)。
# Target是一年后患疾病的定量指标，适合用于回归任务
print(datasets)  # 糖尿病数据
x_data = diabetes.data  # 获取特征数据
# diabetes_data是通过pandas库中的read_csv函数读取名为diabetes.csv的数据文件得到的数据集。
# 这个数据集包含了关于葡萄糖、血压、皮肤厚度、胰岛素、身体质量指数等特征的信息
y_data = diabetes.target  # 获取目标数据
print('特征数据:\n', x_data)
# （442,10）442个数据，每行数据有10个特征数据，相当于y=a1 x1+a2 x2+...a10 x10
print(x_data.shape)
print('目标数据：\n', y_data)
print(y_data.shape)
# 把数据集拆分成：训练集和测试集
from sklearn.model_selection import train_test_split  # 用于将数据集拆分为训练集和测试集。 train训练，test测试，split拆分
from sklearn.linear_model import LinearRegression  # 用于线性回归的模型

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2,
                                                    random_state=9)  # 将特征数据和目标数据拆分成训练集和测试集，
# 其中测试集占总数据的20%，random_state参数用于指定随机种子，保证每次运行代码得到的结果是一致的。
clf = LinearRegression().fit(x_train, y_train)  # 创建一个LinearRegression对象并使用训练数据对其进行拟合。
print('模型的信息:\n', clf)  # 打印模型的信息
# 预测测试集结果
y_pred = clf.predict(x_test)  # 使用训练好的模型对测试集进行预测
print('预测结果为：', '\n', y_pred[:20])  # 打印预测结果的前20个值
from sklearn.metrics import explained_variance_score, mean_absolute_error, \
    mean_squared_error, median_absolute_error, r2_score  # 导入用于评估预测结果的指标函数

print('平均绝对误差：', mean_absolute_error(y_test, y_pred))
print('均方误差：', mean_squared_error(y_test, y_pred))
print('中值绝对误差:', median_absolute_error(y_test, y_pred))
print('可解释方差值：', explained_variance_score(y_test, y_pred))
print('R方值：', r2_score(y_test, y_pred))
