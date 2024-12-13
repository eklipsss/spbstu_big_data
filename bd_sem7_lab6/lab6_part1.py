import copy
import matplotlib.pyplot as plt
import scipy.stats as sps
import numpy as np
import math
from sklearn.linear_model import LinearRegression


def linear_regression(X_, y_):
    X = np.array(X_)
    y = np.array(y_)
    reg = LinearRegression()
    reg.fit(X, y)
    coef = reg.coef_
    y_predict = reg.predict(X)
    return coef, y_predict


def RSS_RSE(y, y_predict, p):
    n = len(y)
    RSS = sum([(y[k] - y_predict[k]) ** 2 for k in range(n)])
    RSE = math.sqrt(RSS/(n-p-1))
    return RSS, RSE


def eta_sqr(y, y_predict):
    return sps.pearsonr(y, y_predict).statistic


def draw_table(X, y, y_predict):
    A = []
    for k in range(len(y)):
        row = list(X[k]) + [y[k], y_predict[k]]
        row = [round(value, 9) for value in row]
        A.append(row)

    name_column = ['x_1', 'x_2', 'x_3', 'y', 'y_predict']
    print('-' * (15*5 + 16))
    print('| {:^15} | {:^15} | {:^15} | {:^15} | {:^15} |'.format(*name_column))
    print('-' * (15*5 + 16))
    for row in A:
        print('| {:^15} | {:^15} | {:^15} | {:^15} | {:^15} |'.format(*row))
        print('-' * (15*5 + 16))


if __name__ == "__main__":
    size_vector = 20
    factor_vector = [sps.norm.rvs(size=3) for _ in range(size_vector)]
    norm_e = sps.norm.rvs(size=size_vector)

    y_vector = []

    for i in range(size_vector):
        factor = factor_vector[i]
        y_k = 1 + 3 * factor[0] - 2 * factor[1] + factor[2] + norm_e[i]
        y_vector.append(y_k)

    # print(factor_vector)
    # print(y_vector)

    coef, y_predict_ = linear_regression(factor_vector, y_vector)
    coef = [round(c, 9) for c in coef]
    # print(coef)

    print("\n")
    print('Коэффициенты')
    name_column = ['beta_1', 'beta_2', 'beta_3']
    print('-' * (15 * 3 + 10))
    print('| {:^15} | {:^15} | {:^15} |'.format(*name_column))
    print('-'*(15*3 + 10))
    print('| {:^15} | {:^15} | {:^15} |'.format(*coef))
    print('-' * (15 * 3 + 10))
    print("\n")

    draw_table(factor_vector, y_vector, y_predict_)

    RSS, RSE = RSS_RSE(y_vector, y_predict_, 3)
    eta = eta_sqr(y_vector, y_predict_)

    print('\n')
    name_column = ['RSS', 'RSE', 'корр. отн-ие']
    print('-' * (15 * 3 + 10))
    print('| {:^15} | {:^15} | {:^15} |'.format(*name_column))
    print('-' * (15 * 3 + 10))
    row = [round(RSS, 7), round(RSE, 7), round(eta, 7)]
    print('| {:^15} | {:^15} | {:^15} |'.format(*row))
    print('-' * (15 * 3 + 10))
