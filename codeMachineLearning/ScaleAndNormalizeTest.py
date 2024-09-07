import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
'''
    特征缩放：将特征值缩放到一个固定范围，使得算法的输入特征具有相同的尺度，从而提高模型的性能和 stabilize learning process
    
'''
# 生成一些示例数据
np.random.seed(42)
X = np.random.rand(100, 3)  # 100个样本，3个特征
y = 5 * X[:, 0] + 3 * X[:, 1] - 2 * X[:, 2] + np.random.randn(100)  # 线性关系加噪声

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 创建和训练SGDRegressor
sgd_regressor = SGDRegressor(max_iter=1000, tol=1e-3, penalty='l2', alpha=0.0001, random_state=42)
sgd_regressor.fit(X_train_scaled, y_train)

# 获取截距和回归系数
intercept = sgd_regressor.intercept_
coefficients = sgd_regressor.coef_

print(f"截距: {intercept}")
print(f"回归系数: {coefficients}")
