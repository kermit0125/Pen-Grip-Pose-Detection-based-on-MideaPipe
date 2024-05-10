import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
import seaborn as sns
import joblib


df=pd.read_csv('ProcessedData.csv')
X = df.drop('Gesture', axis=1)   # 特征列
y = df['Gesture']               # 标签列

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10) # 划分测试集和训练集，测试集20%，训练集80%

model = RandomForestClassifier(max_depth=5, n_estimators=100, random_state=5)  # 100棵决策树，最大深度5，随机数种子
model.fit(X_train, y_train)

# 计算预测结果和预测概率
y_predict = model.predict(X_test)
y_predict_proba = model.predict_proba(X_test)

# 计算混淆矩阵
cm = confusion_matrix(y_test, y_predict)
print("Confusion Matrix:")
print(cm)

# 计算 Precision、Recall、F1-Score
precision = precision_score(y_test, y_predict)
recall = recall_score(y_test, y_predict)
f1 = f1_score(y_test, y_predict)
print("\nPrecision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)

# 绘制混淆矩阵的热图
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
# 保存混淆矩阵图像
plt.savefig('Confusion_Matrix.png')

# 计算 ROC 曲线
fpr, tpr, thresholds = roc_curve(y_test, y_predict_proba[:, 1])
roc_auc = auc(fpr, tpr)

# 绘制 ROC 曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
# 保存混淆矩阵图像
plt.savefig('ROC.png')


# 将模型保存到文件
joblib.dump(model, 'random_forest_model.pkl')


#以下为模型的使用方法
# 在需要的时候加载模型
#loaded_model = joblib.load('random_forest_model.pkl')  #这里是模型的保存地址

# 使用加载的模型进行预测
#new_predictions = loaded_model.predict(new_data)
#new_probabilities = loaded_model.predict_proba(new_data)

# 输出预测结果和概率
#print("Predictions:", new_predictions)
#print("Probabilities:", new_probabilities)