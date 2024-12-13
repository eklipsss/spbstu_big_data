import matplotlib.pyplot as plt
from matplotlib.cbook import boxplot_stats
import scipy.stats as sps
import numpy as np
import math
import seaborn as sns
import pandas as pd
from tabulate import tabulate


def print_table(matrix, headers):
    print(tabulate(matrix, headers, tablefmt="simple_grid", stralign='center'))


def three_sigma(data_list):
    x_avg = sum(data_list) / len(data_list)
    s = math.sqrt(
        sum([(x - x_avg) ** 2 for x in data_list]) / len(data_list)
    )
    r_limit = x_avg - 3 * s
    l_limit = x_avg + 3 * s
    return r_limit, l_limit


def boxplot_T(data_list):
    LQ = np.quantile(data_list, 0.25)
    UQ = np.quantile(data_list, 0.75)
    IQR = UQ - LQ
    s = sorted(data_list)
    x_L = max(s[0], LQ - 3 / 2 * IQR)
    x_U = min(s[-1], UQ + 3 / 2 * IQR)

    return x_L, x_U, LQ, UQ


def checking_for_anomaly(data_list, r_lim, l_lim, m):
    list_start = [i for i in range(m)]
    list_end = [i for i in range(len(data_list) - m, len(data_list))]
    list_ind_for_check = list_start + list_end

    A = []

    for i in list_ind_for_check:
        row = [f"x({i + 1})", f"{data_list[i]}"]
        if r_lim <= data_list[i] <= l_lim:
            row.append("не аномальна")
        else:
            row.append("аномальна")
        A.append(row)

    print_table(A, headers=['x_i', 'Значение', 'Результат'])


x = sps.norm.rvs(size=195).tolist()

x += [5, -4, 3.3, 2.99, -3]

x = sorted(x)
k_s = [k for k in range(len(x))]

print("---------Правило 3 сигм:---------")
r_l, l_l = three_sigma(x)
print('Нижняя граница = ', r_l, '\nВерхняя граница = ', l_l, '\n')
checking_for_anomaly(x, r_l, l_l, 3)

print("\n---------Боксплот Тьюки:---------")
x_L, x_U, LQ, UQ = boxplot_T(x)
print('Нижняя граница = ', x_L, '\nВерхняя граница = ', x_U, '\n')
checking_for_anomaly(x, x_L, x_U, 3)

# sns.boxplot(data=x, flierprops={"marker": "."}, orient="v", color='violet', label="Системный боксплот")
plt.subplot(1, 2, 1)
plt.scatter(k_s, x, color="red", marker=".")
plt.axhline(y=r_l, color='green', label="Правило 3 сигм")
plt.axhline(y=l_l, color='green')
plt.axhline(y=x_L, color='blue', label="Боксплот Тьюки")
plt.axhline(y=x_U, color='blue')
plt.axhline(y=LQ, color='cyan', label="Квартили боксплота Тьюки")
plt.axhline(y=UQ, color='cyan')
plt.legend()

plt.subplot(1, 2, 2)
sns.boxplot(data=x, flierprops={"marker": "."}, orient="v", color='violet', label="Системный боксплот")
# sns.boxplot(data=x, flierprops={"marker": "."}, orient="v", color='violet')
fliers = boxplot_stats(x).pop(0)['fliers']
fliers = np.sort(fliers)
for flier in fliers:
    plt.axhline(y=flier, color='magenta')

plt.legend()
plt.show()
