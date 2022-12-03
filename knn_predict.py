from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import auc, classification_report, confusion_matrix, plot_confusion_matrix, roc_curve
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

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state = 0)


'''KNN>>>>'''
k_fold_range = [5, 10]
# k_range = [3, 5, 7, 8, 9, 10, 11, 13, 15, 17, 19]

for kfold in k_fold_range:
    mean_error = []; std_error = []
    for k in range(3, 70):
        model = KNeighborsClassifier(n_neighbors = k, weights = 'uniform')
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(model, X, y, cv = kfold, scoring = 'f1')
        mean_error.append(scores.mean())
        std_error.append(scores.std())

    plt.errorbar(range(3, 70), mean_error, yerr=std_error, linewidth=1, label = f'k-fold: {kfold}')
plt.xlabel("K"); plt.ylabel("F1 Score")
plt.legend()
plt.show()
'''<<<<KNN'''

#KNN
model = KNeighborsClassifier(n_neighbors = 27, weights = 'uniform')

model.fit(Xtrain, ytrain)
ypred_knn = model.predict(Xtest)
print(">>>>>>>>>>>>>>>>>>>>>>KNN<<<<<<<<<<<<<<<<<<<<")
print(confusion_matrix(ytest, ypred_knn))
print(classification_report(ytest, ypred_knn))
plot_confusion_matrix(model, Xtest, ytest, display_labels=["y = 0", "y = 1"])
plt.title("KNN Regression Confusion Matrix")
plt.show()
print("-------------------------------------------------------------")
print(f'Accuracy of train: {model.score(Xtrain, ytrain)}')
print(f'Accuracy of test: {model.score(Xtest, ytest)}')
#draw roc curve
fpr_knn, tpr_knn, _knn = roc_curve(ytest,ypred_knn)
print(f'AUC of KNN: {auc(fpr_knn, tpr_knn)}')

#dummy
#baseline model
dummy = DummyClassifier(strategy="most_frequent").fit(Xtrain, ytrain)
ypred_dummy = dummy.predict(Xtest)
print(">>>>>>>>>>>>>>>>>>>>>>Dummy Model<<<<<<<<<<<<<<<<<<<<")
print(confusion_matrix(ytest, ypred_dummy))
print(classification_report(ytest, ypred_dummy))
plot_confusion_matrix(dummy, Xtest, ytest, display_labels=["y = 0", "y = 1"])
plt.title("Most Frequent Classifier Confusion Matrix")
plt.show()
print("-------------------------------------------------------------")
fpr, tpr, _ = roc_curve(ytest,ypred_dummy)
plt.plot(fpr, tpr, label="baseline model", c="green")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.plot([0,1], [0, 1], color = 'black', linestyle = '--')
test_auc = metrics.roc_auc_score(ytest, ypred_dummy)

plt.plot(fpr_knn, tpr_knn, label="knn", c="red")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.plot([0,1], [0, 1], color = 'yellow', linestyle = '--')


plt.show()