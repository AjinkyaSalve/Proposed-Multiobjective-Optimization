# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 21:36:40 2026

@author: Ajinkya Salve
"""
#This repository contains the implementation of the algorithm presented in the paper:  
#Multi-Objective Optimization and BHARAT Method Ranking for the Selected Additive, Thermal, and Subtractive Manufac-turing Processes
# Ravipudi Venkata Rao1, Ajinkya Kishor Salve1 and Joao Paulo Davim2
#1-Department of Mechanical Engineering, Sardar Vallabhbhai National Institute of Technology, Surat 395007, India
#2-Department of Mechanical Engineering, University of Aveiro, Campus Santiago, 3810-193 Aveiro, Portugal
# This code implements the ZDT-1 benchmark function evaluated in our study. 
# To execute the algorithm for the other benchmark functions, you may replace the mathematical models (objective functions) and the parameter bounds (search space constraints) directly within the code.

import random
import math
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymoo.problems import get_problem
from pymoo.indicators.igd import IGD
from pymoo.indicators.hv import HV
import time
from datetime import datetime

warnings.filterwarnings("ignore")

class Solution:
    def __init__(self, variables):
        self.variables = variables
        self.objective_values = None
        self.crowding_distance = 0.0

# Initialize the Population
def initialize_population(population_size, num_variables, variable_ranges):
    population = []
    for _ in range(population_size):
        variables = [random.uniform(low, high) for low, high in variable_ranges]
        population.append(Solution(variables))
    return population

# Evaluate the objective function values (Fitness calculation)
def evaluate_objectives(solution, objective_functions):
    solution.objective_values = [func(solution.variables) for func in objective_functions]
    return solution.objective_values

# Dominance relationship check
def dominates(objective_values1, objective_values2, minimization=True):
    better_or_equal = True
    strictly_better = False
    for i in range(len(objective_values1)):
        if minimization:
            if objective_values1[i] > objective_values2[i]:
                better_or_equal = False
                break
            if objective_values1[i] < objective_values2[i]:
                strictly_better = True
        else: # Maximization
            if objective_values1[i] < objective_values2[i]:
                better_or_equal = False
                break
            if objective_values1[i] > objective_values2[i]:
                strictly_better = True
    return better_or_equal and strictly_better

# Archive Update
def update_archive(archive, solution, archive_size, minimization=True):
    non_dominated = True
    new_archive = []
    for archived_solution in archive:
        if dominates(solution.objective_values, archived_solution.objective_values, minimization):
            continue
        elif dominates(archived_solution.objective_values, solution.objective_values, minimization):
            non_dominated = False
            new_archive.append(archived_solution)
        else:
            new_archive.append(archived_solution)

    if non_dominated:
        new_archive.append(solution)
    if len(new_archive) > archive_size:
        new_archive = apply_niching(new_archive, archive_size)
    return new_archive

# Crowding distance computation
def calculate_crowding_distance(archive):
    if len(archive) <= 2:
        for sol in archive:
            sol.crowding_distance = float('inf')
        return
    num_objectives = len(archive[0].objective_values)
    for sol in archive:
        sol.crowding_distance = 0.0
    for m in range(num_objectives):
        archive.sort(key=lambda sol: sol.objective_values[m])
        archive[0].crowding_distance = float('inf')
        archive[-1].crowding_distance = float('inf')
        f_min = archive[0].objective_values[m]
        f_max = archive[-1].objective_values[m]
        if f_max == f_min:
            continue
        for i in range(1, len(archive) - 1):
            archive[i].crowding_distance += (archive[i + 1].objective_values[m] - archive[i - 1].objective_values[m]) / (f_max - f_min)
# Niching update
def apply_niching(archive, archive_size):
    calculate_crowding_distance(archive)
    archive.sort(key=lambda sol: sol.crowding_distance, reverse=True)
    return archive[:archive_size]

# Proposed MOO update rule
def multi_objective_proposed(population_size, num_variables, variable_ranges, objective_functions, max_iterations, archive_size, minimization=True):
    population = initialize_population(population_size, num_variables, variable_ranges)
    archive = []
    archives_over_time = []

    for iteration in range(max_iterations):
        for solution in population:
            evaluate_objectives(solution, objective_functions)

        current_iter_archive = list(archive)
        for solution in population:
            current_iter_archive = update_archive(current_iter_archive, solution, archive_size, minimization)
        archive = current_iter_archive

        for sol in archive: # Ensure objective values are present before copying
            if sol.objective_values is None:
                evaluate_objectives(sol, objective_functions)
        archives_over_time.append([sol for sol in archive])

        new_population = []
        for i in range(population_size):
            x_archive_selected = random.choice(archive)
            x_population_random = random.choice(population)
            xi = population[i]
            r1 = random.random()
            r2 = random.random()
            new_variables = [xi.variables[j] + r1 * (x_archive_selected.variables[j] - xi.variables[j]) + r2 * (xi.variables[j] - x_population_random.variables[j]) for j in range(num_variables)]
            new_variables = [max(variable_ranges[j][0], min(variable_ranges[j][1], val)) for j, val in enumerate(new_variables)]
            new_xi = Solution(new_variables)
            evaluate_objectives(new_xi, objective_functions)
            new_population.append(new_xi)
        population = new_population

    return archive, archives_over_time

# ZDT-1 (bi Objective) benchmark function (For the other benchmark functions this block need to be replaced)
def zdt1_f1(x):
    return x[0]

def zdt1_g(x):
    return 1 + 9 * sum(x[1:]) / (len(x) - 1)

def zdt1_f2(x):
    g = zdt1_g(x)
    return g * (1 - math.sqrt(zdt1_f1(x) / g))

# Performance metrics computations
def gd_custom(PF, truePF, q=2):
    PF = np.array(PF)
    truePF = np.array(truePF)
    m1 = PF.shape[0]
    m = truePF.shape[0]

    if m1 == 0 or m == 0:
        return np.nan

    max_vals = np.max(truePF, axis=0)
    min_vals = np.min(truePF, axis=0)

    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1

    normalized_PF = (PF - min_vals) / range_vals
    normalized_true_PF = (truePF - min_vals) / range_vals

    GD = 0.0
    for i in range(m1):
        diff = normalized_PF[i] - normalized_true_PF
        dist = np.sqrt(np.sum(diff**2, axis=1))
        if len(dist) > 0:
            GD += np.min(dist) ** q
        else:
            return np.nan

    GD = (GD ** (1.0 / q)) / m1
    return GD

def igd_custom(PF, truePF, q=2):
    PF = np.array(PF)
    truePF = np.array(truePF)
    m1 = PF.shape[0]
    m = truePF.shape[0]

    if m1 == 0 or m == 0:
        return np.nan

    max_vals = np.max(truePF, axis=0)
    min_vals = np.min(truePF, axis=0)

    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1

    normalized_PF = (PF - min_vals) / range_vals
    normalized_true_PF = (truePF - min_vals) / range_vals

    IGD = 0.0
    for i in range(m):
        diff = normalized_true_PF[i] - normalized_PF
        dist = np.sqrt(np.sum(diff**2, axis=1))
        if len(dist) > 0:
            IGD += np.min(dist) ** q
        else:
            return np.nan

    IGD = (IGD ** (1.0 / q)) / m
    return IGD

def calculate_spacing(pareto_front):
    pareto_front = np.array(pareto_front)
    N = len(pareto_front)
    if N <= 1:
        return np.nan

    sorted_pf = pareto_front[np.argsort(pareto_front[:, 0])]

    distances = []
    for i in range(N):
        dists = [np.linalg.norm(sorted_pf[i] - sorted_pf[j]) for j in range(N) if i != j]
        if dists:
            distances.append(min(dists))
        else:
            return np.nan

    distances = np.array(distances)
    if len(distances) == 0:
        return np.nan

    mean_d = np.mean(distances)

    if N > 1 and np.sum((distances - mean_d) ** 2) < 1e-12:
        return 0.0

    spacing = np.sqrt(np.sum((distances - mean_d) ** 2) / (N - 1))
    return spacing

def calculate_spread(true_pf, algorithm_data):
    if len(algorithm_data) <= 1 or len(true_pf) == 0:
        return np.nan

    true_pf = np.array(true_pf)
    algorithm_data = np.array(algorithm_data)

    true_pf_sorted = true_pf[np.argsort(true_pf[:, 0])]
    algorithm_data_sorted = algorithm_data[np.argsort(algorithm_data[:, 0])]

    extreme_true_f1 = true_pf_sorted[0]
    extreme_true_f2 = true_pf_sorted[-1]

    df = np.min(np.linalg.norm(algorithm_data_sorted - extreme_true_f1, axis=1))
    dl = np.min(np.linalg.norm(algorithm_data_sorted - extreme_true_f2, axis=1))

    distances = [np.linalg.norm(algorithm_data_sorted[i+1] - algorithm_data_sorted[i]) for i in range(len(algorithm_data_sorted) - 1)]

    if len(distances) == 0:
        return np.nan

    mean_d = np.mean(distances)

    numerator = df + dl + np.sum(np.abs(np.array(distances) - mean_d))
    denominator = df + dl + (len(algorithm_data_sorted) - 1) * mean_d

    if denominator == 0:
        return 0.0 if numerator == 0 else np.nan

    spread = numerator / denominator
    return spread

def calculate_metrics(true_pf_data, algorithm_data, ref_point):
    gd_value, igd_value, hv_value, spacing_value, spread_value = np.nan, np.nan, np.nan, np.nan, np.nan

    if len(algorithm_data) > 0:
        algorithm_data_np = np.array(algorithm_data)

        if len(true_pf_data) > 0:
            true_pf_data_np = np.array(true_pf_data)

            gd_value = gd_custom(algorithm_data_np, true_pf_data_np)
            igd_value = igd_custom(algorithm_data_np, true_pf_data_np)

            spread_value = calculate_spread(true_pf_data_np, algorithm_data_np)

        if ref_point is not None:
            ind_hv = HV(ref_point=np.array(ref_point))
            hv_value = ind_hv(algorithm_data_np)

        spacing_value = calculate_spacing(algorithm_data_np)

    return gd_value, igd_value, hv_value, spacing_value, spread_value


# Main execution(For the other benchmark functions this block need to be replaced )
num_variables = 30
variable_ranges = [(0, 1)] * num_variables
objective_functions = [zdt1_f1, zdt1_f2]
population_size = 100
max_iterations = 400
archive_size = 100
num_runs = 30
reference_point = [1, 1] # Reference point for HV for ZDT1 (objectives are generally <= 1)

problem = get_problem("zdt1")
true_pf = problem.pareto_front()

# Initialize lists to store metrics for each iteration across all runs
all_runs_metrics_per_iter = {
    "GD": [[] for _ in range(max_iterations)],
    "IGD": [[] for _ in range(max_iterations)],
    "HV": [[] for _ in range(max_iterations)],
    "Spacing": [[] for _ in range(max_iterations)],
    "Spread": [[] for _ in range(max_iterations)]
}

final_run_metrics_values = {
    "GD": [], "IGD": [],
    "HV": [], "Spacing": [], "Spread": []
}
run_times = []
start_time_total = time.time()

print("Starting Multi-Objective Proposed Optimization...")

for run in range(num_runs):
    start_time_run = time.time()
    final_pareto_front, archives_per_iteration = multi_objective_proposed(population_size, num_variables, variable_ranges, objective_functions, max_iterations, archive_size)
    end_time_run = time.time()
    run_time = end_time_run - start_time_run
    run_times.append(run_time)

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] Run {run + 1}/{num_runs} completed in {run_time:.4f} seconds.")

    for i, archive_at_iter in enumerate(archives_per_iteration):
        front_values = np.array([sol.objective_values for sol in archive_at_iter if sol.objective_values is not None])
        gd, igd, hv, spacing, spread_value = calculate_metrics(true_pf, front_values, reference_point)

        all_runs_metrics_per_iter["GD"][i].append(gd)
        all_runs_metrics_per_iter["IGD"][i].append(igd)
        all_runs_metrics_per_iter["HV"][i].append(hv)
        all_runs_metrics_per_iter["Spacing"][i].append(spacing)
        all_runs_metrics_per_iter["Spread"][i].append(spread_value)

    final_archive_values = np.array([sol.objective_values for sol in final_pareto_front if sol.objective_values is not None])
    gd, igd, hv, spacing, spread_value = calculate_metrics(true_pf, final_archive_values, reference_point)
    final_run_metrics_values["GD"].append(gd)
    final_run_metrics_values["IGD"].append(igd)
    final_run_metrics_values["HV"].append(hv)
    final_run_metrics_values["Spacing"].append(spacing)
    final_run_metrics_values["Spread"].append(spread_value)


end_time_total = time.time()
total_time = end_time_total - start_time_total
print(f"\nTotal Execution Time for {num_runs} runs: {total_time:.2f} seconds")

# Calculate average metrics over all runs for plotting
avg_metrics_over_time = {
    metric_name: [np.nanmean(values) if values else np.nan for values in metric_lists]
    for metric_name, metric_lists in all_runs_metrics_per_iter.items()
}

# X-axis for convergence plots: Number of function evaluations
function_evaluations = [(i + 1) * population_size for i in range(max_iterations)]


# ==========================================
# PLOT 1: Convergence of Performance Metrics (External Legend Setup)
# ==========================================
plt.figure(figsize=(10.5, 6))

metrics_to_plot_on_single_graph = {
    "GD": "GD",
    "IGD": "IGD",
    "Spacing": "Spacing (SP)",
    "Spread": "Spread (SD)",
    "HV": "Hypervolume (HV)",
}

colors = ['blue', 'green', 'red', 'purple', 'orange'] 
line_styles = ['-', '--', ':', '-.', (0, (3, 5, 1, 5))] 

for i, (metric_key, metric_label) in enumerate(metrics_to_plot_on_single_graph.items()):
    if metric_key in avg_metrics_over_time: 
        plt.plot(function_evaluations, avg_metrics_over_time[metric_key],
                 label=f'{metric_label}',
                 color=colors[i % len(colors)],
                 linestyle=line_styles[i % len(line_styles)],
                 marker='o', markersize=2, linewidth=1.2)

plt.title('Convergence of Performance Metrics for ZDT1 Problem', fontsize=16, fontweight='bold', pad=12)
plt.xlabel('Number of Function Evaluations', fontsize=15, labelpad=8)
plt.ylabel('Metric Value', fontsize=15, labelpad=8)
plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=12, frameon=True, facecolor='white', edgecolor='gray')
plt.grid(False) 
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.tight_layout()
plt.savefig('zdt1convg.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()
print("Saved: Figure_Pareto_Front_Final.png successfully.")

# ==================================================
# PLOT 2: True vs Obtained Pareto Front Validation
# ==================================================
obtained_pareto_front = np.array([sol.objective_values for sol in final_pareto_front if sol.objective_values is not None])

plt.figure(figsize=(7.5, 6))
plt.scatter(true_pf[:, 0], true_pf[:, 1], color='blue', label='True Pareto Front', s=45, zorder=1)
plt.scatter(obtained_pareto_front[:, 0], obtained_pareto_front[:, 1], color='red', label='Obtained Pareto Front', s=20, alpha=0.8, zorder=2)
plt.title('True vs. Obtained Pareto Front for ZDT1', fontsize=16, fontweight='bold', pad=12)
plt.ylabel('$f_2$', fontsize=16)
plt.legend(loc='upper right', fontsize=14, frameon=True, facecolor='white', framealpha=0.9)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.grid(False)
plt.tight_layout()
plt.savefig('zdt1pf.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()
print("Saved: Figure_Pareto_Front_Final.png successfully.")

# Statistics for final run metrics
print("\n--- Performance Metric Statistics ---")
mean_final_metrics = {} # Dictionary to store mean of final metrics for comparison
for metric_name, values in final_run_metrics_values.items():
    valid_values = [v for v in values if not np.isnan(v)]
    if valid_values:
        # For metrics where lower is better (GD, IGD, Spacing, Spread)
        # For HV, higher is better
        if metric_name in ["GD", "IGD", "Spacing", "Spread"]:
            best = np.min(valid_values)
        else: # Assuming HV
            best = np.max(valid_values)
        mean = np.mean(valid_values)
        std_dev = np.std(valid_values)
        mean_final_metrics[metric_name] = mean # Store the mean for later comparison
        print(f"{metric_name}: Best = {best:.8f}, Mean = {mean:.8f}, Std Dev = {std_dev:.8f}")
    else:
        print(f"{metric_name}: No valid values calculated.")

# Add average run time
avg_run_time = np.mean(run_times) if run_times else np.nan
print(f"Average Run Time per run: {avg_run_time:.4f} seconds")
