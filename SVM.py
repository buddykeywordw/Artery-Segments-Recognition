from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import numpy as np
import pandas as pd

# 加载数据
data_A = pd.read_csv('Dense_Features_Position.csv')  # 假设数据文件是CSV格式
X_A = data_A.iloc[:, 1:]  # 特征
y_A = data_A.iloc[:, 0]   # 标签

data_B = pd.read_csv('l32_Features_Position.csv')
X_B = data_B.iloc[:, 1:]  # 特征
y_B = data_B.iloc[:, 0]   # 标签

# 创建SVM模型
svm_quad_A = SVC(kernel='poly', degree=2, probability=True)
svm_quad_B = SVC(kernel='poly', degree=2, probability=True)

# 用StratifiedKFold以保持相同比例的类别分布
skf = StratifiedKFold(n_splits=10)

# 计算A文件的概率预测值
prob_A = cross_val_predict(svm_quad_A, X_A, y_A, cv=skf, method='predict_proba')

# 计算B文件的概率预测值
prob_B = cross_val_predict(svm_quad_B, X_B, y_B, cv=skf, method='predict_proba')

# 取属于正类的概率作为特征
prob_features_A = prob_A[:, 1]
prob_features_B = prob_B[:, 1]

# 创建新的特征数据集，其中包含A和B模型的概率值
prob_features = np.column_stack((prob_features_A, prob_features_B))

# 创建一个新的SVM模型
svm_quad_final = SVC(kernel='poly', degree=2, probability=True)

# 计算融合模型的10-fold交叉验证分数，使用新特征和A的标签
# 这里我们假设A和B的标签是一致的，因此用y_A作为最终的标签数据
final_scores = cross_val_score(svm_quad_final, prob_features, y_A, cv=skf)

# 计算平均准确率
final_accuracy = final_scores.mean()
print("最终SVM模型的准确率（带10-fold CV）：{:.2f}%".format(final_accuracy * 100))