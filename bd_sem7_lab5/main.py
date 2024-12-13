import copy
import math
import statistics as st
import scipy.stats as sps
import numpy as np
import matplotlib.pyplot as plt

from IPython.display import display
import pandas as pd
from tabulate import tabulate


# def print_result_table(dict_distribution_estimator: dict):
#     name_estimator = list((list(dict_distribution_estimator.values())[0]).keys())
#
#     column_name = []
#     for name in name_estimator:
#         column_name.append([name, "среднее"])
#         column_name.append([name, "дисперсия"])
#
#     column_names = pd.DataFrame(column_name,
#                                 columns=["", ""])
#
#     rows = []
#     index = list(dict_distribution_estimator.keys())
#     for value_distribution in list(dict_distribution_estimator.values()):
#         row_val = []
#         for value_estimator_list in list(value_distribution.values()):
#             row_val += value_estimator_list
#         rows.append(row_val)
#
#     columns = pd.MultiIndex.from_frame(column_names)
#
#     df = pd.DataFrame(rows, columns=columns, index=index)
#     pd.set_option('display.max_columns', None)
#     pd.set_option('display.width', None)
#     pd.set_option('display.max_colwidth', None)
#     pd.set_option('display.colheader_justify', 'right')
#     # df.style.format_index(str.upper)
#     #display(df)
#     s = df.to_string().replace('\n', '\n' + '_' * 120 + '\n')
#     print(s)


def print_result_table(dict_distribution_estimator: dict):
    name_estimator = [""] + list((list(dict_distribution_estimator.values())[0]).keys())

    column_name = [""]
    for _ in name_estimator:
        column_name.append("среднее")
        column_name.append("дисперсия")

    A = [column_name]
    index = list(dict_distribution_estimator.keys())
    i = 0
    for value_distribution in list(dict_distribution_estimator.values()):
        row_val = [index[i]]
        for value_estimator_list in list(value_distribution.values()):
            row_val += [round(value, 6) for value in value_estimator_list]
        A.append(row_val)
        i += 1

    print('-' * (23 + 33 * 4 + 10))
    print('| {:^23} | {:^33} | {:^31} | {:^31} | {:^31} |'.format(*name_estimator))
    print('-' * (23 + 33 * 4 + 10))
    for row in A:
        print('| {:^23} | {:^15} | {:^15} | {:^15}| {:^15}| {:^15}| {:^15}| {:^15}| {:^15}|'.format(*row))
        print('-' * (23 + 33 * 4 + 10))
    print()


def boxplot_T(data_list):
    # LQ = np.quantile(data_list, 0.25)
    # UQ = np.quantile(data_list, 0.75)
    UQ, LQ = np.percentile(data_list, [75, 25])
    IQR = UQ - LQ
    s = sorted(data_list)
    x_L = max(s[0], LQ - 1.5 * IQR)
    x_U = min(s[-1], UQ + 1.5 * IQR)

    list_without_emissions = [i for i in s if x_L <= i <= x_U]
    return list_without_emissions


def division_into_three_parts_for_huber(data_list, k, eta):
    n_1 = 0
    n_2 = 0
    list_x_eta = []
    new_data_list = [x if abs(x - eta) <= k
                     else x-k if x - eta > k
                     else x+k
                     for x in data_list]

    for x in new_data_list:
        if abs(x - eta) <= k:
            list_x_eta.append(x)
        elif x - eta > k:
            n_2 += 1
        else:
            n_1 += 1

    return n_1, n_2, list_x_eta, new_data_list


def huber_estimator(data_list, k):
    eta = np.median(data_list)
    while not all(x - eta <= k for x in data_list):
        n_1, n_2, list_x_eta, new_data_list \
            = division_into_three_parts_for_huber(data_list, k, eta)

        data_list = copy.copy(new_data_list)

        eta = (sum(list_x_eta) + (n_2 - n_1)*k) / len(data_list)

    return eta


class Characteristics:
    def __init__(self, n: int, distribution: str, param: dict):
        self.n = n
        self.distribution = distribution
        self.param = param
        self.dict_estimator = {"Mean": [],
                               "Med": [],
                               "Huber": [],
                               "TwoStep": []}

    def generate_sample(self) -> list:
        match self.distribution:
            case "normal":
                return sps.norm.rvs(size=self.n)
            case "cauchy":
                return sps.cauchy.rvs(loc=self.param["loc"], scale=self.param["scale"], size=self.n)
            case "mix":
                return 0.9 * sps.norm.rvs(size=self.n) + \
                    0.1 * sps.cauchy.rvs(loc=self.param["loc"], scale=self.param["scale"], size=self.n)

    def monte_carlo(self, list):
        list_sqr = [i ** 2 for i in list]
        m = sum(list) / len(list)
        d = sum(list_sqr) / len(list_sqr) - m ** 2
        return m, d

    def average_sample(self, sample_list):
        list_avg = []
        for list_x in sample_list:
            # list_x = self.gen_list()
            avg = sum(list_x) / self.n
            list_avg.append(avg)

        m, d = self.monte_carlo(list_avg)
        # list_sqr_avg = [i ** 2 for i in list_avg]
        # m_x_ = sum(list_avg) / len(list_avg)
        # d_x_ = sum(list_sqr_avg) / len(list_sqr_avg) - m_x_ ** 2
        # print(f'среднее среднего выборочного = {m_x_}')
        # print(f'квадрат среднего выборочного = {d_x_}\n')
        self.dict_estimator["Mean"] = [m, d]

    def mediana_sample(self, sample_list):
        list_med = []
        for list_x in sample_list:
            # list_x = self.gen_list()
            med = st.median(list_x)
            list_med.append(med)

        m, d = self.monte_carlo(list_med)
        # list_sqr_med = [i ** 2 for i in list_med]
        # m_med = sum(list_med) / len(list_med)
        # d_med = sum(list_sqr_med) / len(list_sqr_med) - m_med ** 2
        # print(f'среднее медианы = {m_med}')
        # print(f'квадрат медианы = {d_med}\n')
        self.dict_estimator["Med"] = [m, d]

    def hubers_estimator(self, sample_list, k):
        list_h_est = []
        for list_x in sample_list:
            # list_x = self.gen_list()
            huber_est = huber_estimator(list_x, k)
            list_h_est.append(huber_est)

        # list_sqr_H = [i**2 for i in list_H_est]
        # m_H = sum(list_H_est) / len(list_H_est)
        # d_H = sum(list_sqr_H) / len(list_sqr_H) - m_H ** 2
        m, d = self.monte_carlo(list_h_est)

        self.dict_estimator["Huber"] = [m, d]

    def two_step_estimator(self, sample_list):
        list_2_step = []
        for list_x in sample_list:
            # list_x = self.gen_list()
            list_after_boxplot = boxplot_T(list_x)
            avg = sum(list_after_boxplot) / len(list_after_boxplot)
            list_2_step.append(avg)

        m, d = self.monte_carlo(list_2_step)

        # list_sqr_2_step = [i ** 2 for i in list_2_step]
        # m_2_step = sum(list_2_step) / len(list_2_step)
        # d_2_step = sum(list_sqr_2_step) / len(list_sqr_2_step) - m_2_step ** 2
        # print(f'среднее медианы = {m_2_step}')
        # print(f'квадрат медианы = {d_2_step}\n')
        self.dict_estimator["TwoStep"] = [m, d]

    def generate_sample_list(self):
        sample_list = []
        for _ in range(1000):
            sample = self.generate_sample()
            sample_list.append(sample)
        return sample_list

    def call_all_func(self, k):
        sample_list = self.generate_sample_list()
        self.average_sample(sample_list)
        self.mediana_sample(sample_list)
        self.hubers_estimator(sample_list, k)
        self.two_step_estimator(sample_list)


if __name__ == "__main__":
    size_n = 100
    k = 1.44

    param_cauchy = {"loc": 0,
                    "scale": 1}

    normal_distribution = Characteristics(size_n, "normal", {})
    normal_distribution.call_all_func(k)

    cauchy_distribution = Characteristics(size_n, "cauchy", param_cauchy)
    cauchy_distribution.call_all_func(k)

    mix_distribution = Characteristics(size_n, "mix", param_cauchy)
    mix_distribution.call_all_func(k)

    dict_distib_estimator = {"N(0,1)": normal_distribution.dict_estimator,
                             "C(0,1)": cauchy_distribution.dict_estimator,
                             "0.9*N(0,1)+0.1*C(0,1)": mix_distribution.dict_estimator}

    print_result_table(dict_distib_estimator)
