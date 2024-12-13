import copy
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import scipy.stats
import numpy as np
import math

from scipy.stats import norm, chisquare

alpha_list = [0.01, 0.05, 0.1, 0.3]
h = 0.1


def turning_points(list):
    print("  Метод поворотных точек: ")
    X = []
    n = len(list)
    for i in range(n - 2):
        if list[i] < list[i + 1] > list[i + 2]:
            X.append(1)
        elif list[i] > list[i + 1] < list[i + 2]:
            X.append(1)
        else:
            X.append(0)
    p = sum(X)
    E_p = (2 / 3) * (n - 2)
    D_p = (16 * n - 29) / 90
    s_p = math.sqrt(D_p)
    print(f"     p = {p}, E(p) = {E_p}, D(p) = {D_p}, s(p) = {s_p}")
    if abs(p - E_p) < 3 * s_p:
        print("     --> Ряд случаен")
    elif p > E_p:
        print("     --> Ряд является быстро колеблющимся")
    else:
        print("     --> Последовательные значения ряда положительно коррелированы")


def coef_kendall(list):
    print("  Коэффициент Кендалла: ")
    n = len(list)
    P = 0
    for i in range(n):
        for j in range(i, n):
            if list[i] > list[j]:
                P += 1

    tau = (4 * P) / (n * (n - 1)) - 1
    d_tau = 2 * (2 * n + 5) / (9 * n * (n - 1))
    s_tau = math.sqrt(d_tau)
    print(f"     tau = {tau}, D(tau) = {d_tau}, s(tau) = {s_tau}")
    if abs(tau) < 3 * s_tau:
        print("     --> Ряд случаен")
    elif tau > 0:
        print("     --> Наблюдается возрастающий тренд")
    else:
        print("     --> Наблюдается падающий тренд")


def exponential_moving_average(x: list, alpha: float):
    y = [(x[0] + x[1]) / 2]

    for k in range(1, len(x)):
        y_k = alpha * x[k] + (1 - alpha) * y[k - 1]
        y.append(y_k)

    return y


def fourier_amplitude_spectrum(x_k, list_y, k_list):
    N = len(k_list)
    # k_list = [k / N for k in k_list]

    y_fft = fft(x_k)
    y_f = [2.0 / N * np.abs(y) for y in y_fft]

    # color = [(0, 0.392, 0), (0.678, 1, 0.184), (0, 0, 0.502), (1, 0.843, 0, 0.5)]
    # color = ["teal", "hotpink", "gold", "cornflowerblue"]
    color = ["teal", "hotpink", "cornflowerblue", "gold"]

    # plt.plot(k_list[:(N // 2)], y_f[:(N // 2)], color=(1, 0.271, 0), label="Для модельного ряда")
    plt.plot(k_list[:(N // 2)], y_f[:(N // 2)], color="red", label="Для модельного ряда")

    for i in range(len(list_y)):
        y_fft = fft(list_y[i])
        y_f = [2.0 / N * np.abs(y) for y in y_fft]
        plt.plot(k_list[:N // 2], y_f[:N // 2], color=color[i],
                 label=f"Для сглаженного ряда с alpha={alpha_list[i]}")

        # print(f"главная частота для {alpha_list[i]}", fftfreq(y_f))

    plt.title("Амплитудный спектр Фурье")
    plt.xlabel("")
    plt.ylabel("")
    plt.legend()
    plt.show()


def draw(x_k, list_y, list_alpha, k_list):
    # color = [(0, 0.392, 0), (0.678, 1, 0.184), (0, 0, 0.502), (1, 0.843, 0, 0.5)]
    color = ["teal", "hotpink", "cornflowerblue", "gold"]
    for i in range(len(list_y)):
        plt.plot(k_list, list_y[i], color=color[i], label=f"Экспоненциальное скользящее среднее "

                                                          f"с alpha={list_alpha[i]}")
    x_list = [0.5 * math.sin(k * 0.1) for k in k_list]
    plt.plot(k_list, x_list, color="red", label="Тренд")
    plt.scatter(k_list, x_k, color='gray', marker='.', alpha=0.8, label="Модельный ряд")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    normal_error = norm.rvs(size=501)
    k = [k for k in range(501)]
    x_k = [0.5 * math.sin(k * h) + normal_error[k] for k in k]

    Y = []
    R = []

    for alpha in alpha_list:
        y_alpha = exponential_moving_average(x_k, alpha)
        Y.append(y_alpha)
        residue = [x_k[i] - y_alpha[i] for i in range(len(y_alpha))]
        R.append(residue)
        print(f"\nЭкспоненциальное скользящее среднее alpha = {alpha}:")

        print("- Случайность:")
        turning_points(residue)
        coef_kendall(residue)
        print()


        print("- Несмещённость:")
        print("   M[residue] = ", np.mean(residue), end=" => ")
        if round(float(np.mean(residue)), 3) == 0:
            print("оценка несмещённая\n")
        else:
            print("оценка смещённая\n")

        print("- Нормальность по Хи-квадрат:")
        rubbish, p_value = chisquare(residue)
        print("   p_value = ", p_value)
        if p_value > 0.05:
            print("   p_value > 0.05 => остатки нормальны")
        else:
            print("   p_value <= 0.05 => остатки ненормальны")
        print()

    draw(x_k, Y, alpha_list, k)
    # draw(x_k, R, alpha_list, k)
    fourier_amplitude_spectrum(x_k, Y, k)
