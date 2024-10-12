import mlrose_hiive as mlrose
import numpy as np
import time
import matplotlib.pyplot as plt
import random

seed = 42
np.random.seed(seed)

# Helper function to plot fitness vs iteration for a single algorithm
def plot_fitness_vs_iterations(problem_name, algorithm_name, fitness_curve, param_name, param_values):
    plt.figure(figsize=(10, 6))
    for i, curve in enumerate(fitness_curve):
        plt.plot(curve[:, 1], curve[:, 0], label=f"{param_name}: {param_values[i]}")
    plt.title(f"{algorithm_name} Fitness vs Iterations on {problem_name}")
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.legend()
    plt.show()

# Helper function to run an experiment on a given problem using a specific algorithm
def run_experiment(problem, algorithm, params, param_name, max_iters=1000, curve=True):
    fitness_curve_data = []
    best_fitness_data = []
    wall_clock_times = []

    for param in params:
        best_fitness = 0
        for _ in range(5):
            start_time = time.time()

            if algorithm == 'rhc':
                best_state, best_fitness, fitness_curve = mlrose.random_hill_climb(problem, max_iters=max_iters,
                                                                                curve=curve, restarts=param)
            elif algorithm == 'sa':
                schedule = mlrose.ExpDecay(exp_const=param)
                best_state, best_fitness, fitness_curve = mlrose.simulated_annealing(problem, schedule=schedule,
                                                                                    max_iters=max_iters, curve=curve)
            elif algorithm == 'ga':
                best_state, best_fitness, fitness_curve = mlrose.genetic_alg(problem, pop_size=param, max_iters=max_iters,
                                                                            curve=curve)
            elif algorithm == 'mimic':
                best_state, best_fitness, fitness_curve = mlrose.mimic(problem, pop_size=param, max_iters=max_iters,
                                                                    curve=curve)
            
            elapsed_time = time.time() - start_time
            best_fitness = max(best_fitness, best_fitness)
            
            print("The best fitness for trial ", _,' ',best_fitness)
            
        print("Iteration Count", fitness_curve.shape)
        fitness_curve_data.append(fitness_curve)
        best_fitness_data.append(best_fitness)
        wall_clock_times.append(elapsed_time)

    best_fitness_index = np.argmax(best_fitness_data)
    best_fitness_curve = fitness_curve_data[best_fitness_index]

    return fitness_curve_data, best_fitness_curve, best_fitness_data, wall_clock_times

# Function to compare the best parameters across algorithms
def plot_best_algorithms(problem_name, best_fitnesses, algorithms, wall_clock_times, fitness_curves):
    plt.figure(figsize=(10, 6))
    for i, curve in enumerate(fitness_curves):
        plt.plot(curve[:, 1], curve[:, 0], label=f"{algorithms[i]}")

    plt.title(f"Fitness vs Iterations for Best Parameters on {problem_name}")
    plt.xlabel("Iterations")
    plt.ylabel("Fitness")
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(algorithms, best_fitnesses, color='blue')
    plt.title(f"Best Fitness Comparison on {problem_name}")
    plt.ylabel("Best Fitness")
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, round(yval, 2), ha='center', va='bottom')

    plt.show()

    plt.figure(figsize=(10, 6))
    bars = plt.bar(algorithms, wall_clock_times, color='green')
    plt.title(f"Wall Clock Time Comparison on {problem_name}")
    plt.ylabel("Wall Clock Time (seconds)")
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, round(yval, 2), ha='center', va='bottom')

    plt.show()