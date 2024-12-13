import copy
import matplotlib.pyplot as plt
import scipy.stats as sps
import numpy as np
import math
from statistics import median
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def method_turning_points(remained_list):
    print("   Метод поворотных точек: ")
    X = []
    n = len(remained_list)
    for i in range(n - 2):
        if remained_list[i] < remained_list[i + 1] > remained_list[i + 2]:
            X.append(1)
        elif remained_list[i] > remained_list[i + 1] < remained_list[i + 2]:
            X.append(1)
        else:
            X.append(0)
    p = sum(X)
    E_p = (2 / 3) * (n - 2)
    D_p = (16 * n - 29) / 90
    s_p = math.sqrt(D_p)
    print(f"     p = {p}, E(p) = {E_p}, D(p) = {D_p}, s(p) = {s_p}")
    if abs(p - E_p) < 2 * s_p:
        print("     --> Ряд случаен")
    elif p > E_p:
        print("     --> Ряд является быстро колеблющимся")
    else:
        print("     --> Последовательные значения ряда положительно коррелированы")


def coef_kendall(remained_list):
    print("   Коэффициент Кендалла: ")
    n = len(remained_list)
    P = 0
    for i in range(n):
        for j in range(i, n):
            if remained_list[i] > remained_list[j]:
                P += 1
    tau = (4 * P) / (n * (n - 1)) - 1
    d_tau = 2 * (2 * n + 5) / (9 * n * (n - 1))
    s_tau = math.sqrt(d_tau)
    print(f"     tau = {tau}, D(tau) = {d_tau}, s(tau) = {s_tau}")
    if abs(tau) < 2 * s_tau:
        print("     --> Ряд случаен")
    elif tau > 0:
        print("     --> Наблюдается возрастающий тренд")
    else:
        print("     --> Наблюдается падающий тренд")


class Data_for_analyze:
    def __init__(self, size_data: int, h: float):
        self.k_list = [i for i in range(size_data)]
        self.h = h
        norm_e = sps.norm.rvs(size=size_data)
        list_ = [math.sqrt(k * h) + norm_e[k] for k in range(size_data)]
        self.x_list = copy.copy(list_)
        self.moving_average_list = []
        self.moving_median_list = []

    def moving_edge(self, m):
        x_1 = self.k_list[:(2 * m + 1)]
        x_2 = self.k_list[-(2 * m + 1):]

        poly = PolynomialFeatures(degree=3)

        x_poly = poly.fit_transform(np.array(x_1).reshape(-1, 1))
        reg_ = LinearRegression().fit(x_poly, self.x_list[:(2 * m + 1)])

        result_1 = reg_.predict(x_poly)[:m]

        x_poly = poly.fit_transform(np.array(x_2).reshape(-1, 1))
        reg_ = LinearRegression().fit(x_poly, self.x_list[-(2 * m + 1):])

        result_2 = reg_.predict(x_poly)[-m:]

        return list(result_1), list(result_2)

    def moving_average(self, m, window_weight):
        i = 0

        moving_average_list = []

        window_size = 7
        while window_size != window_weight:
            window = self.x_list[:window_size]
            window_median = sum(window) / window_size
            moving_average_list.append(window_median)
            window_size += 2

        while i < len(self.x_list) - window_weight + 1:
            window = self.x_list[i: i + window_weight]
            window_average = sum(window) / window_weight
            moving_average_list.append(window_average)
            i += 1

        while window_size != 7:
            window_size -= 2
            window = self.x_list[-window_size:]
            window_median = sum(window) / window_size
            moving_average_list.append(window_median)

        first_element, last_element = self.moving_edge(3)

        moving_average_list = first_element + moving_average_list + last_element

        self.moving_average_list = copy.copy(moving_average_list)
        return moving_average_list

    def moving_median(self, m, window_weight):

        moving_median_list = [median(sorted([self.x_list[0], self.x_list[1], 3 * self.x_list[1] - 2 * self.x_list[2]]))]

        window_size = 3
        for _ in range(m - 1):
            window = sorted(self.x_list[:window_size])
            window_median = median(window)
            moving_median_list.append(window_median)
            window_size += 2

        i = 0
        while i < len(self.x_list) - window_weight + 1:
            window = sorted(self.x_list[i: i + window_weight])
            window_median = median(window)
            moving_median_list.append(window_median)
            i += 1


        window_size = window_weight - 2
        for _ in range(m - 1):
            window = sorted(self.x_list[-window_size:])
            window_median = median(window)
            moving_median_list.append(window_median)
            window_size -= 2

        moving_median_list.append(median(sorted([self.x_list[-1], self.x_list[-2],
                                                 3*self.x_list[-2] - 2*self.x_list[-3]])))

        self.moving_median_list = copy.copy(moving_median_list)
        return moving_median_list


def draw(k, y_moving_average, y_moving_median, x, i):
    plt.subplot(1, 3, i)
    plt.plot(k, y_moving_average, color="gold")
    plt.plot(k, y_moving_median, color='mediumpurple')
    x_list = [math.sqrt(k * 0.05) for k in range(501)]
    plt.plot(k, x_list, color="red")
    plt.scatter(k, x, color='lightblue', marker=".")

print("\n____________________________________________________________________________")
print("ЗАДАНИЕ 2:")
print("   1. сгенерировать модельный ряд: x_k = sqrt(kh), k = 0,..,500, h=0.05")
print("   2. Выделить тренд методом простого скользящего среднего с шириной окна 21, 51, 111 (m = 10, 25, 55)")
print("   3. Выделить тренд методом простого скользящей медианы с шириной окна 21, 51, 111")
print("   4. Сравнить полученные тренды с точным значением и сделать выводы")
print("   5. Вычесть тренды из ряда и проверить остатки на случайность по числу поворотных точек и коэффициенту Кендела")

k_list = [i for i in range(501)]
h = 0.05
size_ = 501
m_list = [10, 25, 55]
data_analyze = Data_for_analyze(size_, h)
y_average = []
y_median = []

for i in range(len(m_list)):
    m = m_list[i]
    print(f"\n\nm = {m}, ширина окна = 2*m+1 = {2*m+1}")
    y_k_a = data_analyze.moving_average(m, 2 * m + 1)
    y_average.append(y_k_a)

    print("\nСкользящее среднее:")
    x_remained = [data_analyze.x_list[i] - y_k_a[i] for i in range(len(y_k_a))]
    method_turning_points(x_remained)
    coef_kendall(x_remained)

    y_k_m = data_analyze.moving_median(m, 2*m+1)
    y_median.append(y_k_m)
    print("\nСкользящая медиана:")
    x_remained = [data_analyze.x_list[i] - y_k_m[i] for i in range(len(y_k_m))]
    method_turning_points(x_remained)
    coef_kendall(x_remained)
    print()

    draw(k_list, y_k_a, y_k_m, data_analyze.x_list, i+1)

plt.show()
