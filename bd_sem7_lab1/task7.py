import numpy as np
from numpy import inf
from tabulate import tabulate
from task234 import norm_l1, norm_l2, norm_linf

# task 1.7

# eps = 0.001
# l1
# 0.2 0.2 0.2 0.2 0.2
# 0.6 0.1 0.1 0.1 0.1
# l2
# 0.447 0.447 0.447 0.447 0.447
# 0.44 0.44 0.44 0.44 0.44


print("\n____________________________________________________________________________")
print("ЗАДАНИЕ 1.7 - вычисление весовых норм")
# print("**для завершения работы программы введите слово exit")


# def print_table(matrix, headers):
#     print(tabulate(matrix, headers, tablefmt="simple_grid", stralign='center'))


def check_weight(l: str, weights, eps):
    check = 0

    for w_i in weights:
        if w_i <= 0:
            print("  (!) Ошибка: элементы вектора весов должны быть положительными")
            return False

    if l == "l1":
        for w_i in weights:
            check += np.abs(w_i)
    elif l == "l2":
        for w_i in weights:
            check += w_i ** 2
        check = np.sqrt(check)
    elif l == "linf":
        for w_i in weights:
            if w_i>1:
                return False
        check = 1

    # print("  sum = ", check)

    if (1.0 < (check - eps)) or (1.0 > (check + eps)):
        return False
    return True


def get_weight(length: int, l: str, eps):
    weights = np.array(input(f"  Введите вектор весов (элементы вектора должны быть положительными) размера {length}: ").split()).astype(float)

    while len(weights) != length:
        print("\n  (!) Ошибка: неверный размер вектора")
        weights = np.array(input("  Попробуйте снова: ").split()).astype(float)

    while not check_weight(l, weights, eps):
        print("\n  (!) Ошибка: вектор не удовлетворяет ограничениям")
        weights = np.array(input("  Попробуйте снова: ").split()).astype(float)

    print("\n  Введенный вектор весов: ", weights)

    return weights


# length = 5
length = ""
length_int = 0
while (not length.isdigit()) or (length_int == 0):
    length = input("\n  Введите длину вектора: ")
    if length.isdigit() and length != '0':
        length_int = int(length)
    else:
        print("  (!) Ошибка: необходимо ввести целое положительное число")
length = length_int


vec = np.array(input(f"  Введите вектор длины {length}: ").split()).astype(int)
while len(vec) != length:
    print("\n  (!) Ошибка: неверная длина вектора")
    vec = np.array(input("Try again: ").split()).astype(int)
# vec = np.array([1, 2, 3, 4, 5])
print("\n  Введенный вектор: ", vec)

print("\n  Максимальный элемент вектора: ", max(vec))
print("  Минимальный элемент вектора: ", min(vec))
print("  Сумма элементов вектора: ", sum(vec))

l1 = np.linalg.norm(vec, ord=1)
l2 = np.linalg.norm(vec, ord=2)
l_inf = np.linalg.norm(vec, ord=inf)

eps = float(input("\n  Введите точность epsilon = "))

print("\n  Пространство l1")
weightsl1 = get_weight(length, "l1", eps)

print("\n  Пространство l2")
weightsl2 = get_weight(length, "l2", eps)

print("\n  Пространство l^Inf")
weightslinf = get_weight(length, "linf", eps)

weighted_l1 = norm_l1(vec, weightsl1)
weighted_l2 = norm_l2(vec, weightsl2)
weighted_l_inf = norm_linf(vec, weightslinf)

# weighted_l_inf = "not hehe"

print(f"\n  Нормы: \n    l1 = {l1} \n    l2 = {round(l2, 1)} \n    l^inf = {l_inf}")
print(f"\n  Взвешенные нормы: \n    l1 = {weighted_l1} \n    l2 = {round(weighted_l2, 1)} \n    l^inf = {weighted_l_inf}")

# print("\nTHE RESULT OF CALCULATIONS:")
# output = [
#     ["norm", l1, l2, l_inf],
#     ["weighted norm", weighted_l1, weighted_l2, weighted_l_inf]
# ]

# print_table(output, headers=["l1", "l2", "l^inf"])
