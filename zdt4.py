# -*- coding: utf-8 -*-
"""


@author: Ajinkya Salve
"""

#zdt4
import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymoo.problems import get_problem
from pymoo.indicators.gd_plus import GDPlus
from pymoo.indicators.igd import IGD
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.indicators.hv import HV
import time
from datetime import datetime

class Solution:
    def __init__(self, variables):
        self.variables = variables
        self.objective_values = None
        self.crowding_distance = 0.0

def initialize_population(population_size, num_variables, variable_ranges):
    population = []
    for _ in range(population_size):
        variables = [random.uniform(low, high) for low, high in variable_ranges]
        population.append(Solution(variables))
    return population

def evaluate_objectives(solution, objective_functions):
    solution.objective_values = [func(solution.variables) for func in objective_functions]
    return solution.objective_values

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
        else:
            if objective_values1[i] < objective_values2[i]:
                better_or_equal = False
                break
            if objective_values1[i] > objective_values2[i]:
                strictly_better = True
    return better_or_equal and strictly_better

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

def apply_niching(archive, archive_size):
    calculate_crowding_distance(archive)
    archive.sort(key=lambda sol: sol.crowding_distance, reverse=True)
    return archive[:archive_size]

def multi_objective_rao(population_size, num_variables, variable_ranges, objective_functions, max_iterations, archive_size, true_pf, reference_point, minimization=True):
    population = initialize_population(population_size, num_variables, variable_ranges)
    archive = []
    
    
    history_evals = []
    history_gd = []
    history_igd = []
    history_sp = []
    history_sd = []
    history_hv = []
    
    total_evaluations = 0

   
    for solution in population:
        evaluate_objectives(solution, objective_functions)
        total_evaluations += 1
        archive = update_archive(archive, solution, archive_size, minimization)

    for iteration in range(max_iterations):
        new_population = []
        for i in range(population_size):
            x_archive_selected = random.choice(archive)
            x_population_random = random.choice(population)
            xi = population[i]
            r1 = random.random()
            r2 = random.random()
            new_variables = [xi.variables[j] + r1 * (x_archive_selected.variables[j] - xi.variables[j]) + r2 * (xi.variables[j] - x_population_random.variables[j]) for j in range(num_variables)]
            new_variables = [max(0.0, min(1.0, val)) for val in new_variables]
            new_xi = Solution(new_variables)
            evaluate_objectives(new_xi, objective_functions)
            total_evaluations += 1
            new_population.append(new_xi)
            
        
        for solution in new_population:
            archive = update_archive(archive, solution, archive_size, minimization)
        population = new_population
        
        
        front_values = np.array([sol.objective_values for sol in archive])
        gd, _, igd, _, hv, spacing, _, spread_value = calculate_metrics(true_pf, front_values, reference_point)
        
        history_evals.append(total_evaluations)
        history_gd.append(gd)
        history_igd.append(igd)
        history_sp.append(spacing)
        history_sd.append(spread_value)
        history_hv.append(hv)

    return archive, (history_evals, history_gd, history_igd, history_sp, history_sd, history_hv)

def zdt4_f1(x):
    return x[0]

def zdt4_g(x):
    return 1 + 10 * (len(x) - 1) + sum([xi**2 - 10 * math.cos(4 * math.pi * xi) for xi in x[1:]])

def zdt4_f2(x):
    g = zdt4_g(x)
    return g * (1 - math.sqrt(zdt4_f1(x) / g))

def gd_custom(PF, truePF, q=2):
    PF = np.array(PF)
    truePF = np.array(truePF)
    m1 = PF.shape[0]
    
    if m1 == 0: return np.nan

    max_vals = np.max(truePF, axis=0)
    min_vals = np.min(truePF, axis=0)

    normalized_PF = (PF - min_vals) / np.maximum(max_vals - min_vals, 1e-8)
    normalized_true_PF = (truePF - min_vals) / np.maximum(max_vals - min_vals, 1e-8)

    GD = 0.0
    for i in range(m1):
        diff = normalized_PF[i] - normalized_true_PF
        dist = np.sqrt(np.sum(diff**2, axis=1))
        GD += np.min(dist) ** q

    GD = (GD ** (1.0 / q)) / m1
    return GD

def igd_custom(PF, truePF, q=2):
    PF = np.array(PF)
    truePF = np.array(truePF)
    m1 = PF.shape[0]
    m = truePF.shape[0]
    
    if m1 == 0: return np.nan

    max_vals = np.max(truePF, axis=0)
    min_vals = np.min(truePF, axis=0)

    normalized_PF = (PF - min_vals) / np.maximum(max_vals - min_vals, 1e-8)
    normalized_true_PF = (truePF - min_vals) / np.maximum(max_vals - min_vals, 1e-8)

    IGD = 0.0
    for i in range(m):
        diff = normalized_true_PF[i] - normalized_PF
        dist = np.sqrt(np.sum(diff**2, axis=1))
        IGD += np.min(dist) ** q

    IGD = (IGD ** (1.0 / q)) / m
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
    true_pf_norm = (true_pf - true_pf.min(axis=0)) / np.maximum(true_pf.max(axis=0) - true_pf.min(axis=0), 1e-8)
    algorithm_data_norm = (algorithm_data - algorithm_data.min(axis=0)) / np.maximum(algorithm_data.max(axis=0) - algorithm_data.min(axis=0), 1e-8)
    min_distances = np.min(np.linalg.norm(algorithm_data_norm[:, None, :] - true_pf_norm[None, :, :], axis=-1), axis=1)
    coverage = np.mean(min_distances)
    return coverage

def calculate_metrics(true_pf_data, algorithm_data, ref_point):
    gd_value, gd_plus_value, igd_value, igd_plus_value, hv_value, spacing_value, coverage_value, spread_value = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    if len(algorithm_data) > 0 and len(true_pf_data) > 0:
        gd_value = gd_custom(algorithm_data, true_pf_data)
        igd_value = igd_custom(algorithm_data, true_pf_data)
        ind_gd_plus = GDPlus(true_pf_data)
        ind_igd_plus = IGDPlus(true_pf_data)
        ind_hv = HV(ref_point=ref_point)
        gd_plus_value = ind_gd_plus(algorithm_data)
        igd_plus_value = ind_igd_plus(algorithm_data)
        hv_value = ind_hv(algorithm_data)
        spacing_value = calculate_spacing(algorithm_data)
        coverage_value = calculate_coverage(true_pf_data, algorithm_data)
        spread_value = calculate_spread(true_pf_data, algorithm_data)
    elif len(algorithm_data) > 0:
        spacing_value = calculate_spacing(algorithm_data)
    return gd_value, gd_plus_value, igd_value, igd_plus_value, hv_value, spacing_value, coverage_value, spread_value


# Main execution
num_variables = 10
variable_ranges = [(0, 1)] + [(-5, 5)] * (num_variables - 1)
objective_functions = [zdt4_f1, zdt4_f2]
population_size = 100
max_iterations = 400
archive_size = 100
num_runs = 30
reference_point = [1, 1]

problem = get_problem("zdt4")
true_pf = problem.pareto_front()

metrics = {"GD": [], "GD+": [], "IGD": [], "IGD+": [], "HV": [], "Spacing": [], "Coverage": [], "Spread": [], "Run Time": []}
start_time_total = time.time()


for run in range(num_runs):
    start_time_run = time.time()
    pareto_front, convergence_history = multi_objective_rao(
        population_size, num_variables, variable_ranges, objective_functions, 
        max_iterations, archive_size, true_pf, reference_point
    )
    end_time_run = time.time()
    run_time = end_time_run - start_time_run
    
    front_values = np.array([sol.objective_values for sol in pareto_front])
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

end_time_total = time.time()
total_time = end_time_total - start_time_total
print(f"\nTotal Execution Time: {total_time:.2f} seconds")



obtained_pareto_front = np.array([sol.objective_values for sol in pareto_front if sol.objective_values is not None])

plt.figure(figsize=(7.5, 6))

plt.scatter(true_pf[:, 0], true_pf[:, 1], color='blue', label='True Pareto Front', s=45, zorder=1)
plt.scatter(obtained_pareto_front[:, 0], obtained_pareto_front[:, 1], color='red', label='Obtained Pareto Front', s=20, alpha=0.8, zorder=2)

plt.title('True vs. Obtained Pareto Front for ZDT4', fontsize=16, fontweight='bold', pad=12)
plt.xlabel('$f_1$', fontsize=16)
plt.ylabel('$f_2$', fontsize=16)

plt.legend(loc='upper right', fontsize=14, frameon=True, facecolor='white', framealpha=0.9)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.grid(False)
plt.tight_layout()
plt.savefig('zdt4pf.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()
print("Saved: zdt4pf.png successfully.")



history_evals, history_gd, history_igd, history_sp, history_sd, history_hv = convergence_history

plt.figure(figsize=(10.5, 6))

plt.plot(history_evals, history_gd, color='blue', linestyle='-', marker='o', markersize=2, label='GD', zorder=5)
plt.plot(history_evals, history_igd, color='green', linestyle='--', marker='d', markersize=2, label='IGD', zorder=4)
plt.plot(history_evals, history_sp, color='red', linestyle=':', marker='^', markersize=2, label='Spacing (SP)', zorder=3)
plt.plot(history_evals, history_sd, color='purple', linestyle='-.', marker='s', markersize=2, label='Spread (SD)', zorder=2)
plt.plot(history_evals, history_hv, color='orange', linestyle='--', marker='x', markersize=2.5, label='Hypervolume (HV)', zorder=1)
plt.title('Convergence of Performance Metrics for ZDT4 Problem', fontsize=16, fontweight='bold', pad=12)
plt.xlabel('Number of Function Evaluations', fontsize=15, labelpad=8)
plt.ylabel('Metric Value', fontsize=15, labelpad=8)
plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=12, frameon=True, facecolor='white', edgecolor='gray')
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.grid(False)

plt.tight_layout()
plt.savefig('zdt4_convergence.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()
print("Saved: zdt4_convergence.png successfully.")


print("\n--- Performance Metric Statistics ---")
for metric_name, values in metrics.items():
    valid_values = [v for v in values if not np.isnan(v)]
    if valid_values:
        if metric_name not in ["HV"]:
            best = np.min(valid_values)
        else:
            best = np.max(valid_values)
        mean = np.mean(valid_values)
        std_dev = np.std(valid_values)
        print(f"{metric_name}: Best = {best:.8f}, Mean = {mean:.8f}, Std Dev = {std_dev:.8f}")
    else:
        print(f"{metric_name}: No valid values calculated.")