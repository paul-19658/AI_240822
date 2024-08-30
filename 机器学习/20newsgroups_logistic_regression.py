from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer # sklearn包里的用于特征提取的工具feature_extraction，这里是文本的，还有其他类型的比如图像等等
from sklearn.linear_model import LogisticRegression
# 加载20newsgroups数据集
newgroups_train = fetch_20newsgroups(subset='train')
newgroups_test = fetch_20newsgroups(subset='test')

# 创建一个pipeline，用于文件特征提取，接着使用逻辑回归
pipeline = make_pipeline(CountVectorizer(), LogisticRegression(max_iter=3000))

print(newgroups_test.target)

# 训练模型
pipeline.fit(newgroups_train.data, newgroups_train.target)

# 测试模型
y_pred = pipeline.predict(newgroups_test.data)
# print(y_pred)
# 输出模型准确率
print('Accuracy: %.2f' % accuracy_score(newgroups_test.target, y_pred))


