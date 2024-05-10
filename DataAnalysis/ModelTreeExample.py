import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 #pip install scikit-learn导入sklearn包
from sklearn.model_selection import train_test_split        #导入划分训练测试集功能的APA
from sklearn.ensemble import RandomForestClassifier         #导入随机森林模型的APA
from sklearn.tree import export_graphviz                    #导入森林模型Tree可视化工具
from IPython.display import Image
import graphviz

df = pd.read_csv('ProcessedData.csv')

X = df.drop('Gesture',axis=1)   #特征列
y = df['Gesture']               #标签列

X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10) #划分测试集和训练集，测试集80%，训练集20%

model = RandomForestClassifier(max_depth=5, n_estimators=100, random_state=5)           #100棵决策树，最大深度5，随机数种子
model.fit(X_train, y_train)

estimator = model.estimators_[20]
print(estimator)

feature_names =X_train.columns
y_train_str =y_train.astype('str')
y_train_str[y_train_str=='0'] = 'Wrong'
y_train_str[y_train_str=='1'] = 'Correct'
y_train_str=y_train_str.values

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'

dot_file = 'tree.dot'
export_graphviz(estimator, out_file=dot_file,
                feature_names=feature_names,
                class_names=y_train_str,
                rounded=True, proportion=True,
                label='root',
                precision=2, filled=True)

graph = graphviz.Source.from_file(dot_file)
graph.render(filename='tree', format='png', view=True)