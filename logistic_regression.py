import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import plot_confusion_matrix, roc_curve

# def read_data():
# load dataset
df = pd.read_csv('dataset.csv', sep=",", dtype=float)

x_ = []
for i in range(0, df.shape[1]):
    if i != df.shape[1] - 1:
        x_.append(df.iloc[:, i])
    # else:
    #     y = df.iloc[:, i]

X = np.column_stack(tuple(x_))
y = df.iloc[:, 7]
    # return X, y


def plot_logistic_regression_q_selection():
    Ci_range = [0.001, 0.01, 0.1]
    q_range = [1, 2, 3, 4]

    for Ci in Ci_range:
        mean_error = []
        std_error = []
        for q in q_range:
            X_poly = PolynomialFeatures(q).fit_transform(X)
            classifier = LogisticRegression(penalty='l2', C=Ci, max_iter = 1000)
            classifier.fit(X_poly, y)
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(classifier, X_poly, y, cv=5, scoring='f1')
            mean_error.append(np.array(scores).mean())
            std_error.append(np.array(scores).std())
        plt.errorbar(q_range, mean_error, yerr=std_error, linewidth=3, label="Ci % f" % Ci)
    plt.xlabel('q'); plt.ylabel('F1 Score')
    plt.legend()
    plt.title("Different F1 score for different q")
    plt.show()


def plot_logistic_regression_C_selection():
    Ci_range = [0.00001, 0.0001, 0.001, 0.01, 0.1]
    mean_error = []
    std_error = []
    X_poly = PolynomialFeatures(2).fit_transform(X)

    for Ci in Ci_range:
        classifier = LogisticRegression(penalty='l2', C=Ci, max_iter = 1000)
        classifier.fit(X_poly, y)
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(classifier, X_poly, y, cv=5, scoring='f1')
        mean_error.append(np.array(scores).mean())
        std_error.append(np.array(scores).std())
    plt.errorbar(Ci_range, mean_error, yerr=std_error, linewidth=3)
    plt.xlabel('C'); plt.ylabel('F1 Score')
    plt.title("Different F1 score for different C")
    plt.show()


def plot_model_predictions(q, classifier, title):
    X_poly = X
    Xtrain, Xtest, ytrain, ytest = train_test_split(X_poly, y, test_size=0.2)
    plt.scatter(x=Xtest[ytest == 1, 2], y=Xtest[ytest == 1, 6], marker='o', color='green', label='y = 1')
    plt.scatter(x=Xtest[ytest == 0, 2], y=Xtest[ytest == 0, 6], marker='o', color='blue', label='y = 0')

    classifier.fit(Xtrain, ytrain)
    print(Xtrain.shape)
    print(Xtest.shape)

    x_min, y_min = Xtest[:, 2].min(), Xtest[:, 6].min()
    x_max, y_max = Xtest[:, 2].max(), Xtest[:, 6].max()
    # xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
    # Xtest = np.c_[xx.ravel(), yy.ravel()]
    # Xtest = PolynomialFeatures(q).fit_transform(Xtest)

    z = classifier.predict(Xtest)

    new_cmap = mpl.colors.ListedColormap(["purple", "pink"])
    # plt.pcolormesh(xx, yy, z.reshape(xx.shape), shading='auto', cmap=new_cmap, alpha=0.5)
    # plt.xlim(xx.min(), xx.max())
    # plt.ylim(yy.min(), yy.max())

    # purple_patch = mpatches.Patch(color='purple', label='y_pred = +1')
    # pink_patch = mpatches.Patch(color='pink', label='y_pred = -1')
    handles, labels = plt.gca().get_legend_handles_labels()
    # handles.append(purple_patch)
    # handles.append(pink_patch)
    plt.legend(handles = handles, loc='lower left')
    plt.xlabel('jobTitle_software developer');
    plt.ylabel('experience')
    plt.title(f"{title}")
    plt.show()
    X_m1 = X[np.where(y == 0)]
    X_p1 = X[np.where(y == 1)]
    plt.scatter(X_m1[:, 0], X_m1[:, 1], c='r', marker='+', label="y = 0")
    plt.scatter(X_p1[:, 0], X_p1[:, 1], c='b', marker='+', label="y = 1")
    plt.gca().set(title="Training data visualisation", xlabel="X1", ylabel="X2")
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.show()



def plot_confusion_matrix_for_models(q, model, title):
    X_poly = PolynomialFeatures(q).fit_transform(X)
    Xtrain, Xtest, ytrain, ytest = train_test_split(X_poly, y, test_size=0.2)
    model.fit(Xtrain, ytrain)
    plot_confusion_matrix(model, Xtest, ytest, display_labels=["y = 0", "y = 1"])
    plt.title(f"{title}")
    plt.show()
    print("The score of classifier:",
          model.score(Xtest, ytest))


def plot_roc_curve():
    most_frequent_classifier = DummyClassifier(strategy="most_frequent")
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)
    lr_classifier = LogisticRegression(penalty='l2', C=0.01, max_iter=1000)
    most_frequent_classifier.fit(Xtrain, ytrain)
    # draw logistic regression classfier's roc curve
    lr_classifier.fit(Xtrain, ytrain)
    fpr_lr, tpr_lr, threshold_lr = roc_curve(ytest, lr_classifier.decision_function(Xtest))
    from sklearn.metrics import auc
    roc_auc_lr = auc(fpr_lr, tpr_lr)
    plt.plot(fpr_lr, tpr_lr, color='black', linestyle='--',
             label='Logistic Regression Classifier(q=1,C=10) AUC = %0.2f' % roc_auc_lr)

    # draw most frequent classifier's roc curve
    y_scores_mf = most_frequent_classifier.predict_proba(Xtest)
    fpr_mf, tpr_mf, threshold_mf = roc_curve(ytest, y_scores_mf[:, 1])
    roc_auc_mf = auc(fpr_mf, tpr_mf)
    plt.plot(fpr_mf, tpr_mf, 'r', label='Most Frequent Classifier AUC = %0.2f' % roc_auc_mf)

    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('ROC Curve For Models')
    plt.show()


# X, y = read_data()
plot_logistic_regression_q_selection() #q=1
plot_logistic_regression_C_selection() #C choose 0.01
# lr_classifier_1 = LogisticRegression(penalty='l2', C=0.01, max_iter = 1000)
# plot_model_predictions(1, lr_classifier_1, "Logistic Regression Model, q = 1")
# lr_classifier_2 = LogisticRegression(penalty='l2', C=1, max_iter=1000)
# plot_model_predictions(2, lr_classifier_2, "Logistic Regression Model, q = 2")
lr_classifier_c_10 = LogisticRegression(penalty='l2', C=0.01, max_iter=1000)
plot_confusion_matrix_for_models(1, lr_classifier_c_10, "Logistic Regression Confusion Matrix, q = 1, C = 0.01")

most_frequent_classifier = DummyClassifier(strategy="most_frequent")
plot_confusion_matrix_for_models(1, most_frequent_classifier, "Most Frequent Classifier Confusion Matrix")
plot_roc_curve()