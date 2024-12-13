import numpy as np
from numpy import inf

# import matplotlib.pyplot as plt
# import pandas as pd
# from matplotlib.patches import Ellipse
# import scipy.stats as stats
# import math


# task 1.2

x_arr = []
y_arr = []

for i in range(-10, 6):
    x_arr.append(i)

for i in range(-5, 11):
    y_arr.append(i)

x = np.array(x_arr)
y = np.array(y_arr)


# task 1.3

def make_new_vec(x, y):
    z = np.array([0] * len(x))
    for i in range(0, len(x) + 1, 2):
        if i == 0:
            z[i] = x[i]
        elif i == 16:
            z[i - 1] = y[i - 1]
        else:
            z[i] = x[i]
            z[i - 1] = y[i - 1]
    return z


def make_new_vec_2(x, y):
    z = np.array([0] * (len(x) + len(y)))
    for i in range(len(x)):
        z[2 * i] = x[i]
        z[2 * i + 1] = y[i]
    return z


# task 1.4

def norm_l1(z, w):
    s = 0
    for i in range(len(z)):
        s += np.abs(z[i]) * w[i]
    return s


def norm_l2(z, w):
    s = 0
    for i in range(len(z)):
        s += w[i] * z[i] ** 2

    return np.sqrt(s)


def norm_linf(z, w):
    np.abs(z)
    for i in range(len(z)):
        z[i] *= w[i]
    return max(z)


def calculate_norm(z):
    weight = [1]*len(z)

    # print("\n----> Calculations for vec = ", z)
    print("\n  норма в l^1:")
    print("     numpy linalg.norm: ", np.linalg.norm(z, ord=1))
    print("     my func norm_l1: ", norm_l1(z, weight))

    print("\n  норма в l^2:")
    print("     numpy linalg.norm: ", np.linalg.norm(z, ord=2))
    print("     my func norm_l2: ", norm_l2(z, weight))

    print("\n  норма в l^inf:")
    print("     numpy linalg.norm: ", np.linalg.norm(z, ord=inf))
    print("     my func norm_linf: ", norm_linf(z, weight))


if __name__ == "__main__":
    print("\n____________________________________________________________________________")
    print("ЗАДАНИЕ 1.2 - создание векторов x и y\n")

    print("вектор х: ", x, "\nвектор y: ", y)

    print("\n____________________________________________________________________________")
    print("ЗАДАНИЕ 1.3 - получение нового вектора z с помощью векторов x и y\n")

    z1 = make_new_vec(x, y)
    z2 = make_new_vec_2(x, y)

    print("  1) получение вектора z, используя только нечетные элементы x и четные элементы у")
    print("     z1:", z1)
    z1.sort()
    print("     z1 после сортировки:", z1)

    print("\n  2) получение вектора z, используя все элементы x и у")
    print("     z2:", z2)
    z2.sort()
    print("     z2 после сортировки:", z2)

    print("\n____________________________________________________________________________")
    print("ЗАДАНИЕ 4 - вычисление нормы вектора z в пространствах l1, l2, l^inf")

    print("\n  1) z1: ", z1)
    calculate_norm(z1)
    print("\n  2) z2: ", z2)
    calculate_norm(z2)



