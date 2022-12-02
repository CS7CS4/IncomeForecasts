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
# x11 = df.iloc[:, i]
# x12 = df.iloc[:, 1]
# x13 = df.iloc[:, 2]
# x21 = df.iloc[:, 3]
# x22 = df.iloc[:, 4]
# x23 = df.iloc[:, 5]
# x24 = df.iloc[:, 6]
# x31 = df.iloc[:, 7]
# y = df.iloc[:, 8]
X = np.column_stack(tuple(x_))

# poly = PolynomialFeatures(degree = 2, include_bias = False, interaction_only = False)
# Xpoly = poly.fit_transform(X)
# Xpoly_df = pd.DataFrame(Xpoly, columns=poly.get_feature_names_out())

# Xtrain, Xtest, ytrain, ytest = train_test_split(Xpoly_df, y, test_size=0.2, random_state=0)

'''KNN>>>>'''
# kfold = 5
# k_range = [3, 5, 7, 8, 9, 10, 11, 13, 15, 17, 19]

# mean_error = []; std_error = []
# for k in k_range:
#     model = KNeighborsClassifier(n_neighbors = k, weights = 'uniform')
#     from sklearn.model_selection import cross_val_score
#     scores = cross_val_score(model, X, y, cv = kfold, scoring = 'f1')
#     mean_error.append(scores.mean())
#     std_error.append(scores.std())

# plt.errorbar(k_range, mean_error, yerr=std_error, linewidth=1, label = f'k-fold: {kfold}')
# plt.xlabel("KNN-k"); plt.ylabel("F1 Score")
# plt.legend()
# plt.show()
'''<<<<KNN'''

#KNN
model = KNeighborsClassifier(n_neighbors = 9, weights = 'uniform')
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.6, random_state = 0)
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
plt.show()