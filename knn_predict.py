from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.dummy import DummyClassifier
import warnings
warnings.filterwarnings("ignore")

#load dataset
df = pd.read_csv('dataset.csv', sep=",", dtype=float)

x_ = []
for i in range(0, df.shape[1]):
    if i != df.shape[1] - 1:
        x_.append(df.iloc[:, i])
    else:
        y = df.iloc[:, i]

X = np.column_stack(tuple(x_))

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.09, random_state = 0)


'''KNN>>>>'''
# best_p = -1
# best_score = 0.0
# best_k = -1
# kfold = 5
# k_range = [3, 5, 7, 8, 9, 10, 11, 13, 15, 17, 19]

# mean_error = []; std_error = []
# for k in range(10, 100):
#     for p in range(1, 100):
#         model = KNeighborsClassifier(n_neighbors = k, weights = 'distance', p = p)
#         model.fit(Xtrain, ytrain)
#         knn_score = model.score(Xtest, ytest)
#         if knn_score > best_score:
#             best_score = knn_score
#             best_k = k
#             best_p = p

# print("best_p = ", best_p)
# print("best_k = ", best_k)
# print("best_score = ", best_score)
'''<<<<KNN'''

#KNN
model = KNeighborsClassifier(n_neighbors = 38, weights = 'distance', p = 1)

model.fit(Xtrain, ytrain)
ypred_knn = model.predict(Xtest)
print(">>>>>>>>>>>>>>>>>>>>>>KNN<<<<<<<<<<<<<<<<<<<<")
print(confusion_matrix(ytest, ypred_knn))
print(classification_report(ytest, ypred_knn))
print("-------------------------------------------------------------")
print(f'Accuracy of train: {model.score(Xtrain, ytrain)}')
print(f'Accuracy of test: {model.score(Xtest, ytest)}')
#draw roc curve
fpr, tpr, _ = roc_curve(ytest,ypred_knn)
print(f'AUC of KNN: {auc(fpr, tpr)}')
plt.plot(fpr, tpr, label="knn", c="red")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.plot([0,1], [0, 1], color = 'yellow', linestyle = '--')

#dummy
#baseline model
dummy = DummyClassifier(strategy="most_frequent").fit(Xtrain, ytrain)
ypred_dummy = dummy.predict(Xtest)
print(">>>>>>>>>>>>>>>>>>>>>>Dummy Model<<<<<<<<<<<<<<<<<<<<")
print(confusion_matrix(ytest, ypred_dummy))
print(classification_report(ytest, ypred_dummy))
print("-------------------------------------------------------------")
fpr, tpr, _ = roc_curve(ytest,ypred_dummy)
plt.plot(fpr, tpr, label="baseline model", c="green")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.plot([0,1], [0, 1], color = 'black', linestyle = '--')
test_auc = metrics.roc_auc_score(ytest, ypred_dummy)


plt.show()