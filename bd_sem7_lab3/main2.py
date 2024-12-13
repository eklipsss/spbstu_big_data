import pandas as pd
from main1 import *


def alpha_selection(x_list):
    h = 0.1
    alpha_list = [k * h for k in range(1, 10)]
    min = np.inf
    alpha_min = 0

    for alpha_ in alpha_list:
        y_list = [sum(x_list[:3]) / len(x_list[:3])]
        for k in range(1, len(x_list)):
            y_k = alpha_ * x_list[k] + (1 - alpha_) * y_list[k - 1]
            y_list.append(y_k)

        list_sqr = [(x_list[k] - y_list[k+1]) ** 2 for k in range(len(y_list)-1)]
        res_sum = sum(list_sqr)
        if res_sum < min:
            min = res_sum
            alpha_min = alpha_
        y_list.clear()

    return alpha_min


def search_coeff_a(x_k: list, m: int):
    X_matrix = []

    n = len(x_k)
    i = m
    while i != n:
        row = []
        for k in range(m, 0, -1):
            row.append(x_k[i - k])
        X_matrix.append(row)

        i += 1

    X_matrix = np.array(X_matrix)

    b_vect = np.array([-x_k[i] for i in range(m, n)])

    a_coef = np.linalg.lstsq(X_matrix, b_vect, rcond=None)[0]

    return a_coef


def search_h_k(x_list, z_list, m):
    Z_matrix = []
    for k in range(len(x_list)):
        row = []
        for i in range(m):
            row.append(pow(z_list[i], k - 1))
        Z_matrix.append(row)

    Z_matrix = np.array(Z_matrix)
    b_vect = np.array(x_list)

    h_k = np.linalg.lstsq(Z_matrix, b_vect, rcond=None)[0]

    return h_k


def method_proni(x_list, m):
    # 1-ый этап
    a_coef = search_coeff_a(x_list, m)
    # print(a_coef)
    # 2-ой этап
    coef_ = np.insert(a_coef, 0, 1)
    # print(coef_)
    z_k = np.roots(coef_)
    # print(z_k)

    lambda_k_t = [np.log(np.abs(z)) for z in z_k]  # коэффициент затухания
    omega_k_t = [math.atan(z.imag / z.real) / (2 * np.pi) for z in z_k]  # частоты

    # 3-ий этап процедуры
    h_k = search_h_k(x_list, z_k, m)

    A_k = [np.abs(h) for h in h_k]  # амплитуды
    phi_k = [math.atan(h.imag / h.real) for h in h_k]  # фазы

    x_proni = [
        sum([A_k[i] * np.exp(complex(-lambda_k_t[i] * k, omega_k_t[i] * k + phi_k[i])) for i in range(m)])
        for k in range(len(x_list))
    ]
    # print(x_proni)
    return x_proni
    # return A_k, omega_k_t


def check_residue(residue):
    print("\n1. Случайность:")
    # print("\n\t1.1. Метод поворотных точек: ")
    turning_points(residue)
    # print("\n\t1.2. Коэффициент Кендалла: ")
    coef_kendall(residue)

    print("\n2.Несмещённость:\n\tM[residue] = ", np.mean(residue), end=" => ")
    if round(float(np.mean(residue)), 3) == 0:
        print("Оценка несмещённая")
    else:
        print("Оценка смещённая")

    print("\n3. Нормальность по Хи-квадрат:")
    rubbish1, p_value1 = chisquare(residue)
    print("\tp_value = ", p_value1)
    if p_value1 > 0.05:
        print("\tp_value > 0.05 => остатки нормальны")
    else:
        print("\tp_value <= 0.05 => остатки ненормальны")


def part1():
    normal_error = norm.rvs(size=501)
    k1 = [k for k in range(0, 501)]
    x = [0] * len(k1)
    h1 = 0.02
    for i in range(1, len(k1) + 1):
        for k in range(1, 4):
            x[i - 1] += k * math.exp(-(h * i) / k) * math.cos(4 * math.pi * k * h * i + math.pi / k)

    alpha = alpha_selection(x)
    # print("alpha = ", alpha)
    y_k = exponential_moving_average(x, alpha)

    residue = [x[i] - y_k[i] for i in range(len(y_k))]

    print("Первая часть - заданный модельный ряд")
    print(f"  Экспоненциальное скользящее среднее alpha = {alpha}")

    check_residue(residue)

    # plt.subplot(2, 1, 1)
    # plt.scatter(x_list, temperature_list, color='#233d4d', marker='.', label='Ряд температур')
    # plt.plot(x_list, y_k, color='#fe7f2d', label=f'Экспоненциальное среднее alpha={alpha_}')
    # plt.subplot(2, 1, 2)
    # plt.scatter(k1, x1, color='#233d4d', marker='.', label='Ряд из задания номер 1')
    # plt.plot(k1, y_k1, color='#fe7f2d', label=f'Экспоненциальное среднее alpha={alpha1_}')
    # plt.legend()
    # plt.show()

    # y_proni = method_proni(y_k, 3)
    #
    # k_list = [k / (len(x_list) // 2) for k in x_list]
    # x_pr = [el.real for el in y_proni]
    # plt.plot(k_list, x_pr, color="hotpink")
    # plt.title("Метод Прони")
    # plt.show()


def part2():
    data = pd.read_csv('average_daily_temperature_2021_2022.csv')
    date_list = data['Date'].to_list()
    temperature_list = data['Average_daily_temp'].to_list()

    alpha = alpha_selection(temperature_list)
    y_k = exponential_moving_average(temperature_list, alpha)

    residue = [temperature_list[i] - y_k[i] for i in range(len(y_k))]

    print("Температурный ряд среднесуточных температур спб 2021-2022")
    print(f"  Экспоненциальное скользящее среднее alpha = {alpha}")

    check_residue(residue)

    print("\n\n")

    x_list = [k for k in range(len(date_list))]

    plt.scatter(x_list, temperature_list, color='gray', marker='.', label='Ряд температур')
    plt.plot(x_list, y_k, color='red', label=f'Экспоненциальное скользящее среднее alpha={alpha}')
    plt.legend()
    plt.show()

    y_proni = method_proni(temperature_list, 3)

    k_list = [k / (len(x_list) // 2) for k in x_list]
    x_pr = [el.real for el in y_proni]
    plt.plot(k_list, x_pr, color="hotpink")
    plt.title("Метод Прони")
    plt.show()


if __name__ == "__main__":
    print("ЛАБОРАТОРНАЯ РАБОТА №3")
    print("--------------------------------------------------------")
    # part1()
    part2()