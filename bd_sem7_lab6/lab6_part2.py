import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

from lab6_part1 import eta_sqr


def polinomial_regression(y_list, degree):
    x_list = [i for i in range(len(y_list))]
    poly = PolynomialFeatures(degree=degree)
    x_poly = poly.fit_transform(np.array(x_list).reshape(-1, 1))
    # print("\n\nx_f_poly:", x_poly)
    reg_ = LinearRegression().fit(x_poly, y_list)
    # coefs = [coef for coef in reg_.coef_]

    return reg_.predict(x_poly)


def draw_polinomial_regression(x_list, y_list, dict_predict):

    plt.figure(figsize=(13, 7))
    plt.scatter(x_list, y_list, color='red', marker='.', label="Ряд среднегодовой температуры")

    for i in range(1, max_degree):
        plt.plot(x_list, dict_predict[i], label=f"Полином степени = {i}")

    plt.title(f"")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    data = pd.read_csv('average_year_temperature_Severodvinsk.csv')
    # data = pd.read_csv('average_year_temperature_Spb.csv')
    # print(data)

    date_list = data['год'].to_list()
    print("Date = ", date_list)

    temperature_list = data['температура за год'].to_list()
    print("Temperature = ", temperature_list)

    max_degree = 10
    dict_predict_res = {}
    for degree in range(1, max_degree + 1):
        y_predict = polinomial_regression(temperature_list, degree)
        dict_predict_res[degree] = y_predict

    draw_polinomial_regression(date_list, temperature_list, dict_predict_res)

    name_column = ['Степень полинома', 'Корреляционное отношение']
    print('-' * (25 + 30 + 7))
    print('| {:^25} | {:^30} |'.format(*name_column))
    print('-' * (25 + 30 + 7))
    for degree in range(1, max_degree + 1):
        eta = eta_sqr(temperature_list, dict_predict_res[degree])
        row = [degree, round(eta, 7)]
        print('| {:^25} | {:^30} |'.format(*row))
        print('-' * (25 + 30 + 7))
