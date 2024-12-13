from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
from tabulate import tabulate


def print_table(matrix, headers):
    print(tabulate(matrix, headers, tablefmt="simple_grid", stralign='center'))


def RidgeRegression(list_x, list_y, lambda_):
    poly = PolynomialFeatures(degree=11)
    x_poly = poly.fit_transform(np.array(list_x).reshape(-1, 1))

    ridge = Ridge(alpha=lambda_, max_iter=10000)
    reg_ = ridge.fit(x_poly, list_y)

    return reg_, [reg_.intercept_] + reg_.coef_


def LassoRegression(list_x, list_y, lambda_):
    poly = PolynomialFeatures(degree=11)
    x_poly = poly.fit_transform(np.array(list_x).reshape(-1, 1))

    lasso = Lasso(alpha=lambda_, max_iter=10000)
    reg_ = lasso.fit(x_poly, list_y)

    return reg_, [reg_.intercept_] + reg_.coef_


def apply_Ridge_and_Lasso(x, y, lambda_list):
    dict_ridge_reg = {}
    dict_ridge_coef = {}
    for lmbda in lambda_list_:
        y_ridge, coef = RidgeRegression(x, y, lmbda)
        dict_ridge_reg[lmbda] = y_ridge
        dict_ridge_coef[lmbda] = coef

    # Lasso regression
    dict_lasso_reg = {}
    dict_lasso_coef = {}
    for lmbda in lambda_list_:
        y_lasso, coef = LassoRegression(x, y, lmbda)
        dict_lasso_reg[lmbda] = y_lasso
        dict_lasso_coef[lmbda] = coef

    return dict_ridge_reg, dict_lasso_reg, dict_ridge_coef, dict_lasso_coef


def RSS_RSE(y, y_predict, p):
    n = len(y)
    RSS = sum([(y[k] - y_predict[k]) ** 2 for k in range(n)])
    # RSE = math.sqrt(RSS/(n-p-1))
    return RSS


def eta_sqr(y, y_predict):
    return sps.pearsonr(y, y_predict).statistic


def draw_table(x, y, dict_ridge, dict_lasso_):
    name_col = ["", 'Ridge', 'Lasso']
    print('-' * (17 + 65*2 + 4))
    print('| {:^15} | {:^63} | {:^63} |'.format(*name_col))

    name_column = ['Lambda', 'RSS', 'Корреляционное отношение', 'RSS', 'Корреляционное отношение']
    print('-' * (17 + 32*4 + 6))
    print('| {:^15} | {:^30} | {:^30} | {:^30} | {:^30} |'.format(*name_column))
    print('-' * (17 + 32*4 + 6))

    poly = PolynomialFeatures(degree=11)
    x_poly = poly.fit_transform(np.array(x).reshape(-1, 1))

    for lmbda, y_ridge in dict_ridge.items():
        RSS_ridge = RSS_RSE(y, y_ridge.predict(x_poly), 11)
        eta_ridge = eta_sqr(y, y_ridge.predict(x_poly))

        RSS_lasso = RSS_RSE(y, dict_lasso_[lmbda].predict(x_poly), 11)
        eta_lasso = eta_sqr(y, dict_lasso_[lmbda].predict(x_poly))

        row = [lmbda, RSS_ridge, eta_ridge, RSS_lasso, eta_lasso]
        print('| {:^15} | {:^30} | {:^30} | {:^30} | {:^30} |'.format(*row))
        print('-' * (17 + 32*4 + 6))


def draw_table2(x, y, dict_ridge, dict_lasso_):
    name_col = ["", 'Ridge', 'Lasso']
    print('-' * (17 + 65*2 + 4))
    print('| {:^15} | {:^63} | {:^63} |'.format(*name_col))

    name_column = ['Lambda', 'RSS', 'Корреляционное отношение', 'RSS', 'Корреляционное отношение']
    print('-' * (17 + 32*4 + 6))
    print('| {:^15} | {:^30} | {:^30} | {:^30} | {:^30} |'.format(*name_column))
    print('-' * (17 + 32*4 + 6))

    poly = PolynomialFeatures(degree=11)
    x_poly = poly.fit_transform(np.array(x).reshape(-1, 1))

    for lmbda, y_ridge in dict_ridge.items():
        RSS_ridge = RSS_RSE(y, y_ridge.predict(x_poly), 11)
        eta_ridge = eta_sqr(y, y_ridge.predict(x_poly))

        RSS_lasso = RSS_RSE(y, dict_lasso_[lmbda].predict(x_poly), 11)
        eta_lasso = eta_sqr(y, dict_lasso_[lmbda].predict(x_poly))

        row = [lmbda, RSS_ridge, eta_ridge, RSS_lasso, eta_lasso]
        print('| {:^15} | {:^30} | {:^30} | {:^30} | {:^30} |'.format(*row))
        print('-' * (17 + 32*4 + 6))


def draw_graphics(x, y, dict_ridge, dict_lasso, suptitle_):
    plt.figure()
    plt.suptitle(suptitle_)
    plt.subplot(1, 2, 1)
    plt.scatter(x, y, color="red")

    x_linsp = np.linspace(min(x), max(x), 100)
    poly = PolynomialFeatures(degree=11)
    x_poly_100 = poly.fit_transform(np.array(x_linsp).reshape(-1, 1))

    for lmbda, y_ridge_reg in dict_ridge.items():
        plt.plot(x_linsp, y_ridge_reg.predict(x_poly_100), label=f"Lambda = {lmbda}")

    plt.title('Ridge regression')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(x, y, color="red")
    for lmbda, y_lasso_reg in dict_lasso.items():
        plt.plot(x_linsp, y_lasso_reg.predict(x_poly_100), label=f"Lambda = {lmbda}")
    plt.title('Lasso regression')
    plt.legend()

    plt.show()


def draw_table_coef(matrix_coeff_ridge, matrix_coeff_lasso):
    name_col = ["", 'Ridge', 'Lasso']
    print('-' * (17 + 71 * 2 + 4))
    print('| {:^10} | {:^69} | {:^69} |'.format(*name_col))

    A_ridge = np.array(matrix_coeff_ridge).transpose()
    A_lasso = np.array(matrix_coeff_lasso).transpose()
    name_column = ['Coef', 'Без шума', 'N(0, 0.01)', 'N(0, 0.04)', 'N(0, 0.09)',
                   'Без шума', 'N(0, 0.01)', 'N(0, 0.04)', 'N(0, 0.09)']
    print('-' * (17 * 9 + 10))
    print('| {:^10} | {:^15} | {:^15} | {:^15} | {:^15} | {:^15} | {:^15} | {:^15} | {:^15} |'.format(*name_column))
    print('-' * (17 * 9 + 10))

    for i in range(len(A_ridge)):
        row_1 = [c_g for c_g in A_ridge[i]]
        row_2 = [c_l for c_l in A_lasso[i]]
        row = [f"b_{i}"] + row_1 + row_2
        # print('-' * (17 * 9 + 10))
        print('| {:^10} | {:^15} | {:^15} | {:^15} | {:^15} | {:^15} | {:^15} | {:^15} | {:^15} |'.format(*row))
        print('-' * (17 * 9 + 10))


def draw_table_coef_separate(matrix_coeff_ridge, matrix_coeff_lasso):
    name_column = ['Coef', 'Без шума', 'N(0, 0.01)', 'N(0, 0.04)', 'N(0, 0.09)']

    header_Ridge = ["", 'Ridge']
    print('-' * (32 * 4 + 18))
    print('| {:^10} | {:^129} |'.format(*header_Ridge))
    A_ridge = np.array(matrix_coeff_ridge).transpose()
    print('-' * (32 * 4 + 18))
    print('| {:^10} | {:^30} | {:^30} | {:^30} | {:^30} |'.format(*name_column))
    print('-' * (32 * 4 + 18))

    for i in range(len(A_ridge)):
        row_1 = [f"b_{i}"] + [c_g for c_g in A_ridge[i]]
        print('| {:^10} | {:^30} | {:^30} | {:^30} | {:^30} |'.format(*row_1))
        print('-' * (32 * 4 + 18))

    header_Lasso = ["", 'Lasso']
    print('-' * (32 * 4 + 18))
    print('| {:^10} | {:^129} |'.format(*header_Lasso))
    A_lasso = np.array(matrix_coeff_lasso).transpose()
    print('-' * (32 * 4 + 18))
    print('| {:^10} | {:^30} | {:^30} | {:^30} | {:^30} |'.format(*name_column))
    print('-' * (32 * 4 + 18))

    for i in range(len(A_ridge)):
        row_2 = [f"b_{i}"] + [c_l for c_l in A_lasso[i]]
        print('| {:^10} | {:^30} | {:^30} | {:^30} | {:^30} |'.format(*row_2))
        print('-' * (32 * 4 + 18))



def draw_table_coef_separate2(matrix_coeff_ridge, matrix_coeff_lasso):
    name_column = ['Coef', 'Без шума', 'N(0, 0.01)', 'N(0, 0.04)', 'N(0, 0.09)']

    print("________________________________RIDGE_______________________________")
    A_ridge = np.array(matrix_coeff_ridge).transpose()
    ridge_output = [0] * len(A_ridge)

    for i in range(len(A_ridge)):
        row_1 = [f"b_{i}"] + [float(c_g) for c_g in A_ridge[i]]
        ridge_output[i] = row_1
    print_table(ridge_output, headers=name_column)

    print("________________________________LASSO_______________________________")
    A_lasso = np.array(matrix_coeff_lasso).transpose()
    lasso_output = [0] * len(A_lasso)

    for i in range(len(A_lasso)):
        row_2 = [f"b_{i}"] + [float(c_l) for c_l in A_lasso[i]]
        lasso_output[i] = row_2
    print_table(lasso_output, headers=name_column)


if __name__ == "__main__":
    x = [-2, -1, 0, 1, 2]
    y = [-7, 0, 1, 2, 9]

    lambda_list_ = [10, 1, 0.1, 0.01, 0.001]

    # для сохранения коэффициентов
    dict_ridge_coefficient = {}
    dict_lasso_coefficient = {}

    for lmbd in lambda_list_:
        dict_ridge_coefficient[lmbd] = []
        dict_lasso_coefficient[lmbd] = []

    # ==========================
    # Без Шума
    dict_ridge, dict_lasso, dict_ridge_coef, dict_lasso_coef = apply_Ridge_and_Lasso(x, y, lambda_list_)

    print("Без добавления шума")

    for lmbd in lambda_list_:
        dict_ridge_coefficient[lmbd].append(dict_ridge_coef[lmbd])

    for lmbd in lambda_list_:
        dict_lasso_coefficient[lmbd].append(dict_lasso_coef[lmbd])

    # draw_table(x, y, dict_ridge, dict_lasso)
    # draw_graphics(x, y, dict_ridge, dict_lasso, "Без шума")

    # ==========================
    # Добавляем шум
    scale_list = [0.1, 0.2, 0.3]
    norm_e_matrix = [sps.norm(scale=2*scale_list[i]).rvs(size=len(y)) for i in range(len(scale_list))]

    for i in range(len(scale_list)):
        y_e = [y[j] + norm_e_matrix[i][j] for j in range(len(y))]

        print(f"\nС шумом = N(0, {round(scale_list[i], 2)})")

        dict_ridge, dict_lasso, dict_ridge_coef, dict_lasso_coef = apply_Ridge_and_Lasso(x, y_e, lambda_list_)

        for lmbd in lambda_list_:
            dict_ridge_coefficient[lmbd].append(dict_ridge_coef[lmbd])

        for lmbd in lambda_list_:
            dict_lasso_coefficient[lmbd].append(dict_lasso_coef[lmbd])

        # draw_table(x, y_e, dict_ridge, dict_lasso)
        # draw_graphics(x, y_e, dict_ridge, dict_lasso, f"С шумом = N(0, {round(scale_list[i], 2)})")

    for lmbda in lambda_list_:
        print(f"\nДля lambda={lmbda}")
        draw_table_coef_separate2(dict_ridge_coefficient[lmbda], dict_lasso_coefficient[lmbda])
