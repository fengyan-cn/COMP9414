import numpy as np
import random
import matplotlib.pyplot as plt


def objective_function(x):
    return x ** 2

def Tabu_search(objective, criterion, tabu_list, alpha):
    current_point = np.random.randn() * 10
    print(current_point)
    tabu_list.append(current_point)
    current_value = objective(current_point)
    best, best_value = current_point, current_value
    for epoch in range(1000):
        neighbour_list = [(best + alpha * np.random.rand()) for _ in range(50)]
        candidate = neighbour_list[0]
        for j in range(1, len(neighbour_list)):
            delta = objective(neighbour_list[j]) - objective(candidate)
            if (neighbour_list[j] not in tabu_list) and (delta < 0):
                candidate = neighbour_list[j]
        best, best_value = candidate, objective(candidate)
        print(best, best_value)
        tabu_list.append(candidate)
        # print(best_value)

    return best_value


alpha = 0.9
bounds = [-10, 10]
criterion = 1e-10
# np.random.seed(55)
tabu_list = []
score = Tabu_search(objective_function, criterion=criterion, tabu_list=tabu_list, alpha=alpha)
print(score)