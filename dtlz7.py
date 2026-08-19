# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 12:11:11 2026

@author: Ajinkya Salve
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 13:33:58 2025

@author: Ajinkya Salve
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from pymoo.indicators.hv import HV
import time
from mpl_toolkits.mplot3d import Axes3D
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions

# --- Objective Functions ---
def g(x):
    m_prime = len(x) - 2
    if m_prime == 0:
        return 0
    return 1 + (9 / m_prime) * np.sum(x[2:])

def h(f1, f2, g_val):
    return 3 - (f1 / (1 + g_val)) * (1 + np.sin(3 * np.pi * f1)) - (f2 / (1 + g_val)) * (1 + np.sin(3 * np.pi * f2))

def f1(x):
    return x[0]

def f2(x):
    return x[1]

def f3(x):
    g_val = g(x)
    return (1 + g_val) * h(f1(x), f2(x), g_val)

# --- Solution Class ---
class Solution:
    def __init__(self, variables):
        self.variables = variables
        self.objective_values = None
        self.crowding_distance = 0.0

# --- Initialization ---
def initialize_population(population_size, num_variables, variable_ranges):
    return [Solution([random.uniform(low, high) for low, high in variable_ranges]) for _ in range(population_size)]

def evaluate_objectives(solution, objective_functions):
    solution.objective_values = [func(solution.variables) for func in objective_functions]
    return solution.objective_values

# --- Dominance and Archive ---
def dominates(obj1, obj2):
    better_or_equal = all(a <= b for a, b in zip(obj1, obj2))
    strictly_better = any(a < b for a, b in zip(obj1, obj2))
    return better_or_equal and strictly_better

def update_archive(archive, solution, archive_size):
    non_dominated = True
    new_archive = []
    for other in archive:
        if dominates(solution.objective_values, other.objective_values):
            continue
        elif dominates(other.objective_values, solution.objective_values):
            non_dominated = False
            new_archive.append(other)
        else:
            new_archive.append(other)
    if non_dominated:
        new_archive.append(solution)
    if len(new_archive) > archive_size:
        calculate_crowding_distance(new_archive)
        new_archive.sort(key=lambda s: s.crowding_distance, reverse=True)
        new_archive = new_archive[:archive_size]
    return new_archive

def calculate_crowding_distance(archive):
    if len(archive) <= 2:
        for sol in archive:
            sol.crowding_distance = float('inf')
        return
    num_obj = len(archive[0].objective_values)
    for sol in archive:
        sol.crowding_distance = 0.0
    for m in range(num_obj):
        archive.sort(key=lambda sol: sol.objective_values[m])
        archive[0].crowding_distance = archive[-1].crowding_distance = float('inf')
        f_min, f_max = archive[0].objective_values[m], archive[-1].objective_values[m]
        if f_max - f_min == 0:
            continue
        for i in range(1, len(archive) - 1):
            archive[i].crowding_distance += (archive[i + 1].objective_values[m] - archive[i - 1].objective_values[m]) / (f_max - f_min)

# --- Multi-Objective Rao Algorithm ---
def multi_objective_rao(pop_size, num_variables, var_ranges, max_iterations, archive_size, objective_functions):
    population = initialize_population(pop_size, num_variables, var_ranges)
    archive = []
    for sol in population:
        evaluate_objectives(sol, objective_functions)
        archive = update_archive(archive, sol, archive_size)
    for _ in range(max_iterations):
        new_population = []
        for i in range(pop_size):
            x1 = random.choice(archive)
            x2 = random.choice(population)
            xi = population[i]
            r1, r2 = random.random(), random.random()
            new_vars = [max(0.0, min(1.0, xi.variables[j] + r1 * (x1.variables[j] - xi.variables[j]) + r2 * (xi.variables[j] - x2.variables[j]))) for j in range(num_variables)]
            new_sol = Solution(new_vars)
            evaluate_objectives(new_sol, objective_functions)
            archive = update_archive(archive, new_sol, archive_size)
            new_population.append(new_sol)
        population = new_population
    return archive

# --- Metrics ---
def calculate_metrics(algorithm_data, ref_point):
    metrics = {}
    F = np.array([sol.objective_values for sol in algorithm_data])
    try:
        metrics["HV"] = HV(ref_point=ref_point)(F)
    except:
        metrics["HV"] = np.nan
    return metrics

# --- Main Parameters ---
num_variables = 22
variable_ranges = [(0, 1)] * num_variables
population_size = 100
max_iterations = 400
archive_size = 100
reference_point = [1, 1, 1]
num_runs = 1
objective_functions = [f1, f2, f3]

# --- Run the Algorithm ---
all_fronts = []
metrics_all = []
start_all = time.time()

for run in range(num_runs):
    start = time.time()
    archive = multi_objective_rao(population_size, num_variables, variable_ranges, max_iterations, archive_size, objective_functions)
    end = time.time()
    front = np.array([sol.objective_values for sol in archive])
    metrics = calculate_metrics(archive, reference_point)
    metrics["Time"] = end - start
    all_fronts.append(front)
    metrics_all.append(metrics)
    print(f"[Run {run+1}] HV: {metrics['HV']:.5f}, Time: {metrics['Time']:.2f} s")

# --- Get True Pareto Front ---
ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)
true_pf = get_problem("dtlz7").pareto_front()

# --- 3D Pareto Plot (Final Run) Styled for Continuous Overlap ---
final_front = all_fronts[-1]
fig = plt.figure(figsize=(8, 6.5))
ax = fig.add_subplot(111, projection='3d')

# Plot True Pareto Front first (Blue points in background layer)
ax.scatter(true_pf[:, 0], true_pf[:, 1], true_pf[:, 2], 
           color='blue', s=30, alpha=0.6, marker='o', label='True Pareto Front', zorder=1)

# Plot Obtained Pareto Front directly on top (Red solid points overlaying perfectly)
ax.scatter(final_front[:, 0], final_front[:, 1], final_front[:, 2], 
           color='red', s=30, alpha=0.8, marker='o', label='Obtained Pareto Front', zorder=2)

# Set Title configurations
plt.title('True vs. Obtained Pareto Front for DTLZ7', fontsize=16, fontweight='bold', pad=12)

# Set LaTeX Axis Labels and precise padding offsets
ax.set_xlabel('$f_1$', fontsize=16, labelpad=5)
ax.set_ylabel('$f_2$', fontsize=16, labelpad=5)
ax.set_zlabel('$f_3$', fontsize=16, labelpad=-1)

# Orientation setup for true front matching
ax.view_init(elev=30, azim=-135)
ax.invert_yaxis()
ax.invert_xaxis()

# Configure Legend Box formatting
ax.legend(loc='upper right', fontsize=14, frameon=True, facecolor='white', framealpha=0.9)

# Adjust standard tick mark label sizing
ax.tick_params(axis='both', which='major', labelsize=13)

# Clear structural plot grid lines
ax.grid(False)

# Render and Export layout at 300 DPI target resolution
plt.tight_layout()
plt.savefig('dtlz7paretofront.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()
print("Saved: dtlz7paretofront.png successfully.")

# --- Summary ---
print("\n--- HV Summary ---")
hv_values = [m["HV"] for m in metrics_all if not np.isnan(m["HV"])]
print(f"Best HV: {np.max(hv_values):.5f}")
print(f"Mean HV: {np.mean(hv_values):.5f}")
print(f"Std Dev HV: {np.std(hv_values):.5f}")