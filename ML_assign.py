
#xuefy ML_assignment
# CANCER_GENE-EXPRESSION_Classification


import sys
import scipy
import numpy
import matplotlib
import pandas
import sklearn

# 加载机器学习包
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
import warnings

# 输入数据集
names = []
for i in range(20531):
    names.append('gene_' + str(i))

dataset = pandas.read_csv(r'F:\pythonProject\TCGA-PANCAN-HiSeq\data.csv',names=names)
labelset = pandas.read_csv(r'F:\pythonProject\TCGA-PANCAN-HiSeq\labels.csv')

# 数据集形式
print(dataset.shape)
print(labelset.shape)
print(dataset.describe())

#input的数据集和标签
X = dataset.values[1:, :]
Y = labelset.values[:, 1]

#样本数据划分训练集和测试集
validation_size = 0.2
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
                                                                                random_state=seed)

# 衡量效果为accuracy
scoring = 'accuracy'

#便于查看，控制警告的输出
warnings.filterwarnings(action='ignore', category=UserWarning)

# 使用4种算法
models = []
models.append(('SVM', SVC()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('LR', LogisticRegression()))
models.append(('NB', GaussianNB()))


# 轮流使用4种算法分类，使用10折交叉验证，评估指标为准确度的平均值和标准差
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle= True)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    modelscore = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(modelscore)


# 算法分类效果比较
fig = plt.figure()
fig.suptitle('algorithms evaluation')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# 选择算法在测试集中进行预测
print('选择一个算法进行预测:')
for name, model in models:
    print(name)

choice = input('选择算法： ')
best = ''
if choice == 'LR':
    best = LogisticRegression()
elif choice == 'CART':
    best = DecisionTreeClassifier()
elif choice == 'NB':
    best = GaussianNB()
elif choice == 'SVM':
    best = SVC()


if best != '':
    best.fit(X_train, Y_train)
    predictions = best.predict(X_validation)
    print(accuracy_score(Y_validation, predictions)) #输出预测的准确性
    print(confusion_matrix(Y_validation, predictions)) #输出混淆矩阵
    print(classification_report(Y_validation, predictions))#输出测试集中的分类效果报告
