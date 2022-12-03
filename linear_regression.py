import math

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from scipy import stats
from sklearn import linear_model
from scipy.stats import norm, skew
import sys
from scipy import stats
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from pandas.plotting import scatter_matrix

def load_data(file_path):
    df = pd.read_csv(file_path)
    features = df[df.columns[:-1]].to_numpy()
    target = df.iloc[:, -1]
    return features, target


# def load_data(path):
    # res = stats.probplot(df_train["Salary"], plot=plt)
    # plt.show()

    # res = stats.probplot(df_train["Salary"], plot=plt)
    # plt.show()

    # sns.distplot(df_train["Salary"], fit=norm)
    # plt.legend(['Normal dist'], loc='best')
    # plt.ylabel('Frequency')
    # plt.title('Salary distribution')
    # plt.show()
    # return df_train


def draw_norm(df_train):
    res = stats.probplot(df_train["Salary"], plot=plt)
    plt.show()

    sns.distplot(df_train["Salary"], fit=norm)
    plt.legend(['Normal dist'], loc='best')
    plt.ylabel('Frequency')
    plt.title('Salary distribution')
    plt.show()
    return df_train


def plot_scatter(df_train):
    plt.figure(figsize=(16, 8))
    plt.scatter(x=df_train['experience_filter'], y=df_train['Salary'])
    plt.ylim = (0, 800000)
    plt.xlabel('experience_feature ')
    plt.ylabel('Salary ')
    plt.title(' experience_feature and Salary')
    plt.show()


def process_data(df_train):
    experience_map = {
        "Entry": 1,
        "Mid": 2,
        "Senior": 3
    }

    # degreed_map = {
    #     "Bachelor": 1,
    #     "Master": 2
    # }

    df_train['experience_filter'] = df_train['experience_filter'].map(experience_map)
    # df_train['education_filter'] = df_train['education_filter'].map(degreed_map)

    local_oh = pd.get_dummies(df_train.location_filter, prefix = 'location')
    job_oh = pd.get_dummies(df_train.job_title_filter, prefix = 'jobTitle')
    # exp_oh = pd.get_dummies(df_train.experience_filter, prefix = 'exp')

    s = MinMaxScaler()
    df_train["Salary"] = stats.boxcox(df_train["Salary"])[0]
    df_train['Salary'] = s.fit_transform(np.array(df_train['Salary']).reshape(-1, 1))
    # draw_norm(df_train)
    output = pd.concat([job_oh, local_oh, df_train['experience_filter'],  df_train['Salary']], axis = 1)
    output.to_csv("dataset.csv", index=False)
    return output


def reletade(df_train):
    k = 2
    corrmat = df_train.corr()
    # f, ax = plt.subplots(figsize=(12, 9))
    # sns.heatmap(corrmat, vmax=.8, square=True)

    cols_10 = corrmat.nlargest(9, 'Salary')['Salary'].index
    corrs_10 = df_train[cols_10].corr()
    plt.figure(figsize=(12, 9))
    sns.heatmap(corrs_10, annot=True)

    # attributes = ["experience_filter", "job_title_filter", "location_filter", "Salary"]
    # scatter_matrix(df_train[attributes], figsize=(12, 8))
    plt.show()


def plot_errorbar(title, Ci_array, mean_error, std_error):
    plt.errorbar(Ci_array, mean_error, yerr=std_error)
    plt.xlabel('Ci')
    plt.ylabel('Mean square error')
    # plt.xlim((0, 100))

    image_name = '%s.png' % (title)

    if os.path.exists(image_name):
        os.remove(image_name)

    plt.savefig(image_name)
    plt.show()

def custom_loss(y_train, y_predict):

    if np.nan in y_predict:
        print("y_predict has nan")
    elif np.nan in y_train:
        print("y_train has nan")

    loss = np.sqrt(mean_squared_error(np.log(y_train), np.log(y_predict)))
    return loss

def get_mse(y_train, y_predict):
    # rmse = np.sqrt(mean_squared_error(y_train, y_predict))
    mse = mean_squared_error(y_train, y_predict)
    return mse

def r2(y_train, y_predict):
    loss = np.sqrt(mean_squared_error(y_train, y_predict))


def print_baseline(X_train, Y_train, X_test, Y_test, strategy='mean'):
    dummy_reg = DummyRegressor(strategy=strategy)
    dummy_reg.fit(X_train, Y_train)
    # acc = dummy_reg.score(X_test, Y_test)
    # print('baseline acc on the test dataset: {:.2f}'.format(acc))
    y_predict = dummy_reg.predict(X_test)
    rmse = get_mse(Y_test, y_predict)
    print('baseline mse on the test dataset: {:.2f}'.format(rmse))


def augments_feature(features, degree):
    poly = PolynomialFeatures(degree)
    poly_features = poly.fit_transform(features)
    return poly_features


def lasso_cross_validation(Ci_array, d_array, data_path):
    features, target = load_data(data_path)
    for Ci in Ci_array:
        model = train_lasso(Ci, features, target)
        mean_error = []
        std_error = []
        for d in d_array:
            poly_features = augments_feature(features, d)
            # 5 flod cross-validation
            scores = cross_val_score(model, poly_features, target, cv=5, scoring='neg_mean_squared_error')
            mean_error.append(-1 * (np.array(scores).mean()))
            std_error.append(-1 * (np.array(scores).std()))

        plt.errorbar(d_array, mean_error, yerr=std_error, label="C = %.3f" % Ci)
        plt.xlabel('Degree')
        plt.ylabel('Mean Squared Error')

    plt.title("Lasso Cross-Validation")

    image_name = 'lasso_error_bar.png'

    if os.path.exists(image_name):
        os.remove(image_name)

    plt.legend()
    plt.savefig(image_name)
    plt.show()


def ridge_cross_validation(Ci_array, d_array, data_path):
    features, target = load_data(data_path)

    for Ci in Ci_array:
        model = train_ridge(Ci, features, target)
        mean_error = []
        std_error = []
        for d in d_array:
            poly_features = augments_feature(features, d)
            # 5 flod cross-validation
            scores = cross_val_score(model, poly_features, target, cv=5, scoring='neg_mean_squared_error')
            mean_error.append(-1 * (np.array(scores).mean()))
            std_error.append(-1 * (np.array(scores).std()))

        plt.errorbar(d_array, mean_error, yerr=std_error, label="C = %.3f" % Ci)
        plt.xlabel('Degree')
        plt.ylabel('Mean Squared Error')

    plt.title("Ridge Cross-Validation")

    image_name = 'ridge_error_bar.png'

    if os.path.exists(image_name):
        os.remove(image_name)

    plt.legend()
    plt.savefig(image_name)
    plt.show()


def cross_validation_withd_ridge(Ci_array, d, data_path):
    features, target = load_data(data_path)
    poly_features = augments_feature(features, d)
    mean_error = []
    std_error = []

    for Ci in Ci_array:
        model = train_ridge(Ci, poly_features, target)
        # 5 flod cross-validation
        scores = cross_val_score(model, poly_features, target, cv=5, scoring='neg_mean_squared_error')
        # print(scores)
        mean_error.append(-1 * (np.array(scores).mean()))
        std_error.append(-1 * (np.array(scores).std()))

    print(mean_error)
    print(std_error)
    plt.errorbar(Ci_array, mean_error, yerr=std_error)
    plt.xlabel('C')
    plt.ylabel('Mean Squared Error')
    plt.title("Ridge Cross-Validation d=%d" % d)

    image_name = 'Ridge_error_bar_d=%d.png' % d

    if os.path.exists(image_name):
        os.remove(image_name)

    plt.legend()
    plt.savefig(image_name)
    plt.show()


def cross_validation_withd_Lasso(Ci_array, d, data_path):
    features, target = load_data(data_path)
    poly_features = augments_feature(features, d)
    mean_error = []
    std_error = []

    for Ci in Ci_array:
        model = train_lasso(Ci, poly_features, target)
        # 5 flod cross-validation
        scores = cross_val_score(model, poly_features, target, cv=5, scoring='neg_mean_squared_error')
        print(scores)
        mean_error.append(-1*(np.array(scores).mean()))
        std_error.append(-1*(np.array(scores).std()))

    print(mean_error)
    print(std_error)
    plt.errorbar(Ci_array, mean_error, yerr=std_error)
    plt.xlabel('C')
    plt.ylabel('Mean Squared Error')
    plt.title("Lasso Cross-Validation d=%d" % d)

    image_name = 'Lasso_error_bar_d=%d.png' % d

    if os.path.exists(image_name):
        os.remove(image_name)

    plt.legend()
    plt.savefig(image_name)
    plt.show()


def train_lasso(ci, features, target):
    alpha = 1 / (ci * 2)
    X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size=0.2, random_state=0)
    model = linear_model.Lasso(alpha=alpha)
    model.fit(X_train, Y_train)
    y_predict = model.predict(X_test)

    mse = get_mse(Y_test, y_predict)
    # print('lasso mse on the test dataset: {:.2f}'.format(mse))
    # print_baseline(X_train, Y_train, X_test, Y_test)
    return model


def train_ridge(ci, features, target):
    alpha = 1 / (ci * 2)
    X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size=0.2, random_state=0)

    model = linear_model.Ridge(alpha=alpha)
    model.fit(X_train, Y_train)
    y_predict = model.predict(X_test)
    mse = get_mse(Y_test, y_predict)
    # print('ridge mse on the test dataset: {:.2f}'.format(mse))
    # print_baseline(X_train, Y_train, X_test, Y_test)

    return model


if __name__ == "__main__":
    train_data_path = sys.argv[1]
    # df_train = pd.read_csv(train_data_path)
    # df_train = process_data(df_train)
    # reletade(df_train)

    # features, target = load_data(train_data_path)
    ci_array_lasso = [1, 500, 1000, 1500, 2000]
    di_array = [1, 2, 3, 4, 5]
    lasso_cross_validation(ci_array_lasso, di_array, train_data_path)
    cross_validation_withd_Lasso(ci_array_lasso, 4, train_data_path)

    ci_array_ridge = [0.001, 0.01, 1, 10, 20, 30]
    ridge_cross_validation(ci_array_ridge, di_array, train_data_path)
    cross_validation_withd_ridge(ci_array_ridge, 4, train_data_path)




    # df_train = load_data(train_data_path)
    # df_train = pd.read_csv(train_data_path)
    # df_train = process_label_new(df_train)
    # print(set(df_train["experience_filter"]))