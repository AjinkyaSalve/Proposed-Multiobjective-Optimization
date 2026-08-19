# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 14:41:24 2026

@author: Ajinkya Salve
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from pymoo.indicators.hv import HV
from pymoo.indicators.gd import GD
from pymoo.indicators.gd_plus import GDPlus
from pymoo.indicators.igd import IGD
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions
from datetime import datetime
import time
from mpl_toolkits.mplot3d import Axes3D

# --- Utility Functions ---
def clip_variables(variables, lower=0.0, upper=1.0):
    return [max(lower, min(upper, v)) for v in variables]

# --- DTLZ1-like Objectives ---
def g(x):
    return sum((xi - 0.5)**2 for xi in x[2:])

def f1(x):
    return 1 - (1 + g(x)) * np.cos(x[0] * np.pi / 2) * np.cos(x[1] * np.pi / 2)

def f2(x):
    return 1 -  (1 + g(x)) * np.cos(x[0] * np.pi / 2) * np.sin(x[1] * np.pi / 2)

def f3(x):
    return 1 - (1 + g(x)) * np.sin(x[0] * np.pi / 2)

# --- Solution Class ---
class Solution:
    def __init__(self, variables):
        self.variables = variables
        self.objective_values = None
        self.crowding_distance = 0.0

# --- Initialization ---
def initialize_population(population_size, num_variables, variable_ranges):
    return [
        Solution(clip_variables([random.uniform(low, high) for low, high in variable_ranges]))
        for _ in range(population_size)
    ]

def evaluate_objectives(solution, objective_functions):
    solution.objective_values = [func(solution.variables) for func in objective_functions]

# --- Dominance and Archive Management ---
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
        f_min, f_max = archive[0].objective_values[m], archive[-1].objective_values[m]
        archive[0].crowding_distance = archive[-1].crowding_distance = float('inf')
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
            new_vars = [
                xi.variables[j] + r1 * (x1.variables[j] - xi.variables[j]) + r2 * (xi.variables[j] - x2.variables[j])
                for j in range(num_variables)
            ]
            new_vars = clip_variables(new_vars)
            new_sol = Solution(new_vars)
            evaluate_objectives(new_sol, objective_functions)
            archive = update_archive(archive, new_sol, archive_size)
            new_population.append(new_sol)
        population = new_population
    return archive

# --- Metrics ---
def gd_custom(PF, truePF, q=2):
    PF = np.array(PF)
    truePF = np.array(truePF)
    max_vals = np.max(truePF, axis=0)
    min_vals = np.min(truePF, axis=0)
    normalized_PF = (PF - min_vals) / (max_vals - min_vals)
    normalized_true_PF = (truePF - min_vals) / (max_vals - min_vals)
    GD = 0.0
    for i in range(PF.shape[0]):
        diff = normalized_PF[i] - normalized_true_PF
        dist = np.sqrt(np.sum(diff**2, axis=1))
        GD += np.min(dist) ** q
    GD = (GD ** (1.0 / q)) / PF.shape[0]
    return GD

def igd_custom(PF, truePF, q=2):
    PF = np.array(PF)
    truePF = np.array(truePF)
    max_vals = np.max(truePF, axis=0)
    min_vals = np.min(truePF, axis=0)
    normalized_PF = (PF - min_vals) / (max_vals - min_vals)
    normalized_true_PF = (truePF - min_vals) / (max_vals - min_vals)
    IGD = 0.0
    for i in range(truePF.shape[0]):
        diff = normalized_true_PF[i] - normalized_PF
        dist = np.sqrt(np.sum(diff**2, axis=1))
        IGD += np.min(dist) ** q
    IGD = (IGD ** (1.0 / q)) / truePF.shape[0]
    return IGD

def calculate_spacing(pareto_front):
    N = len(pareto_front)
    if N <= 1:
        return np.nan
    distances = []
    for i in range(N):
        dists = [np.linalg.norm(pareto_front[i] - pareto_front[j]) for j in range(N) if i != j]
        distances.append(min(dists))
    distances = np.array(distances)
    mean_d = np.mean(distances)
    spacing = np.sqrt(np.sum((distances - mean_d) ** 2) / (N - 1))
    return spacing

def calculate_spread(true_pf, algorithm_data):
    if len(algorithm_data) <= 1:
        return np.nan
    true_pf = np.array(true_pf)
    algorithm_data = np.array(algorithm_data)
    true_pf = true_pf[np.argsort(true_pf[:, 0])]
    algorithm_data = algorithm_data[np.argsort(algorithm_data[:, 0])]
    df = np.linalg.norm(algorithm_data[0] - true_pf[0])
    dl = np.linalg.norm(algorithm_data[-1] - true_pf[-1])
    distances = [np.linalg.norm(algorithm_data[i+1] - algorithm_data[i]) for i in range(len(algorithm_data) - 1)]
    mean_d = np.mean(distances)
    spread = (df + dl + np.sum(np.abs(distances - mean_d))) / (df + dl + (len(algorithm_data) - 1) * mean_d)
    return spread

def calculate_coverage(true_pf, algorithm_data):
    if len(algorithm_data) == 0 or len(true_pf) == 0:
        return np.nan
    true_pf_norm = (true_pf - true_pf.min(axis=0)) / (true_pf.max(axis=0) - true_pf.min(axis=0))
    algorithm_data_norm = (algorithm_data - algorithm_data.min(axis=0)) / (algorithm_data.max(axis=0) - algorithm_data.min(axis=0))
    min_distances = np.min(np.linalg.norm(algorithm_data_norm[:, None, :] - true_pf_norm[None, :, :], axis=-1), axis=1)
    coverage = np.mean(min_distances)
    return coverage

def normalize_front(front):
    front = np.array(front)
    min_vals = np.min(front, axis=0)
    max_vals = np.max(front, axis=0)
    return (front - min_vals) / (max_vals - min_vals + 1e-12)

def calculate_metrics(true_pf_data, algorithm_data, ref_point):
    gd_value = gd_plus_value = igd_value = igd_plus_value = hv_value = spacing_value = coverage_value = spread_value = np.nan

    if len(algorithm_data) > 0 and len(true_pf_data) > 0:
        # Normalize both fronts to [0, 1]
        normalized_algo = normalize_front(algorithm_data)
        normalized_true = normalize_front(true_pf_data)

        gd_value = gd_custom(normalized_algo, normalized_true)
        igd_value = igd_custom(normalized_algo, normalized_true)
        ind_gd_plus = GDPlus(normalized_true)
        ind_igd_plus = IGDPlus(normalized_true)
        ind_hv = HV(ref_point=[1, 1, 1])

        gd_plus_value = ind_gd_plus(normalized_algo)
        igd_plus_value = ind_igd_plus(normalized_algo)
        hv_value = ind_hv(normalized_algo)
        spacing_value = calculate_spacing(normalized_algo)
        coverage_value = calculate_coverage(normalized_true, normalized_algo)
        spread_value = calculate_spread(normalized_true, normalized_algo)

    return gd_value, gd_plus_value, igd_value, igd_plus_value, hv_value, spacing_value, coverage_value, spread_value

# --- Main Parameters ---
num_variables = 12
variable_ranges = [(0, 1)] * num_variables
population_size = 100
max_iterations = 100
archive_size = 100
reference_point = [1, 1, 1]
objective_functions = [f1, f2, f3]
num_runs = 1

# Generate true Pareto front using pymoo's DTLZ2 for 3D visualization
ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)
true_pf = 1 - get_problem("dtlz2").pareto_front(ref_dirs)

# --- Execution and Logging ---
metrics = {"GD": [], "GD+": [], "IGD": [], "IGD+": [], "HV": [], "Spacing": [], "Coverage": [], "Spread": [], "Run Time": []}
all_fronts = []

start_time_total = time.time()

for run in range(num_runs):
    start_time_run = time.time()
    pareto_front = multi_objective_rao(population_size, num_variables, variable_ranges, max_iterations, archive_size, objective_functions)
    end_time_run = time.time()
    run_time = end_time_run - start_time_run
    front_values = np.array([sol.objective_values for sol in pareto_front])
    all_fronts.append(front_values)
    gd, gd_plus, igd, igd_plus, hv, spacing, coverage, spread_value = calculate_metrics(true_pf, front_values, reference_point)
    metrics["GD"].append(gd)
    metrics["GD+"].append(gd_plus)
    metrics["IGD"].append(igd)
    metrics["IGD+"].append(igd_plus)
    metrics["HV"].append(hv)
    metrics["Spacing"].append(spacing)
    metrics["Coverage"].append(coverage)
    metrics["Spread"].append(spread_value)
    metrics["Run Time"].append(run_time)

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] Run {run + 1}: GD = {gd:.8f}, GD+ = {gd_plus:.8f}, IGD = {igd:.8f}, IGD+ = {igd_plus:.8f}, HV = {hv:.8f}, Spacing = {spacing:.8f}, Coverage = {coverage:.8f}, Spread = {spread_value:.8f}, Run Time = {run_time:.8f} seconds")

# --- 3D Pareto Front Visualization Styled for Continuous Overlap ---
final_front = normalize_front(all_fronts[-1])
normalized_true_pf = normalize_front(true_pf)

fig = plt.figure(figsize=(8, 6.5))
ax = fig.add_subplot(111, projection='3d')

# Plot True Pareto Front first (Blue points in background layer)
ax.scatter(normalized_true_pf[:, 0], normalized_true_pf[:, 1], normalized_true_pf[:, 2], 
           color='blue', s=30, alpha=0.6, marker='o', label='True Pareto Front', zorder=1)

# Plot Obtained Pareto Front directly on top (Red solid points overlaying perfectly)
ax.scatter(final_front[:, 0], final_front[:, 1], final_front[:, 2], 
           color='red', s=30, alpha=0.8, marker='o', label='Obtained Pareto Front', zorder=2)

plt.title('True vs. Obtained Pareto Front for IDTLZ2', fontsize=16, fontweight='bold', pad=12)

ax.set_xlabel('$f_1$', fontsize=16, labelpad=5)
ax.set_ylabel('$f_2$', fontsize=16, labelpad=5)
ax.set_zlabel('$f_3$', fontsize=16, labelpad=-1)

# Orientation setup for true front matching and structural alignment
ax.view_init(elev=30, azim=-135)
ax.invert_yaxis()
ax.invert_xaxis()

ax.legend(loc='upper right', fontsize=14, frameon=True, facecolor='white', framealpha=0.9)
ax.tick_params(axis='both', which='major', labelsize=13)
ax.grid(False)

plt.tight_layout()
plt.savefig('idtlz2paretofront.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

print("Saved: idtlz2paretofront.png successfully.")
print("\nVisualization complete!")

# --- Metrics Summary ---
print("\n--- Performance Metric Statistics ---")
for metric_name, values in metrics.items():
    valid_values = [v for v in values if not np.isnan(v)]
    if valid_values:
        best = np.min(valid_values) if metric_name not in ["HV"] else np.max(valid_values)
        mean = np.mean(valid_values)
        std_dev = np.std(valid_values)
        print(f"{metric_name}: Best = {best:.8f}, Mean = {mean:.8f}, Std Dev = {std_dev:.8f}")
    else:
        print(f"{metric_name}: No valid values calculated.")