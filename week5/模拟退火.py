import numpy as np
import matplotlib.pyplot as plt


def objective_function(x):
    return x ** 2

def simulated_annealing(objective, bounds, final_temp, step_size, original_temp):
    best = np.random.randn() * 10
    best_eval = objective(best)
    curr, curr_eval = best, best_eval
    scores = [best_eval]

    while original_temp > final_temp:
        candidate = curr + np.random.randn() * step_size
        candidate = np.clip(candidate, bounds[0], bounds[1])
        candidate_eval = objective(candidate)
        if candidate_eval < best_eval:
            best, best_eval = candidate, candidate_eval
            scores.append(best_eval)
        if candidate_eval < curr_eval or np.random.randn() < np.exp((curr_eval - candidate_eval) / original_temp):
            curr, curr_eval = candidate, candidate_eval
        original_temp *= 0.99
    return [best, best_eval, scores]

def challenge_function(x, y):
    part_1 = np.sin(np.sqrt(np.abs(x / 2 + (y + 47))))
    part_2 = np.sin(np.sqrt(np.abs(x - y - 47)))
    return -(y + 47) * part_1 - x * part_2



np.random.seed(999)
bounds = [-512, 512]
original_temp = 10
final_temp = 1e-5
step_size = 0.9
plt.figure(figsize=(4, 4))
plt.plot(np.linspace(-10, 10, 1000), objective_function(np.linspace(-10, 10, 1000)), c='r')
plt.show()
best, score, scores = simulated_annealing(objective_function, bounds, final_temp, step_size, original_temp)
print(f"best solution: x = {best}, f(x) = {score}")
