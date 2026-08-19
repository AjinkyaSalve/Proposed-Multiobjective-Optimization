# -*- coding: utf-8 -*-
"""


@author: Ajinkya Salve
"""

#dtlz4
import random
import numpy as np
import matplotlib.pyplot as plt
from pymoo.indicators.hv import HV
from pymoo.indicators.gd import GD as pymoo_GD
from pymoo.indicators.gd_plus import GDPlus
from pymoo.indicators.igd import IGD as pymoo_IGD
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions
from datetime import datetime
import time
from mpl_toolkits.mplot3d import Axes3D

alpha = 1


def g(x):
    """Calculates the g function for DTLZ4 (m=3)."""
    x = np.array(x)
    return np.sum((x[2:] - 0.5)**2)

def f1(x):
    """Calculates the first objective function for DTLZ4 (m=3)."""
    gm = g(x)
    return (1 + gm) * np.cos((x[0]**alpha) * np.pi / 2) * np.cos((x[1]**alpha) * np.pi / 2)

def f2(x):
    """Calculates the second objective function for DTLZ4 (m=3)."""
    gm = g(x)
    return (1 + gm) * np.cos((x[0]**alpha) * np.pi / 2) * np.sin((x[1]**alpha) * np.pi / 2)

def f3(x):
    """Calculates the third objective function for DTLZ4 (m=3)."""
    gm = g(x)
    return (1 + gm) * np.sin((x[0]**alpha) * np.pi / 2)


class Solution:
    def __init__(self, variables):
        self.variables = variables
        self.objective_values = None
        self.crowding_distance = 0.0


def initialize_population(population_size, num_variables, variable_ranges):
    return [Solution([random.uniform(low, high) for low, high in variable_ranges]) for _ in range(population_size)]

def evaluate_objectives(solution, objective_functions):
    solution.objective_values = [func(solution.variables) for func in objective_functions]


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
    return np.mean(min_distances)

def calculate_metrics(true_pf_data, algorithm_data, ref_point):
    gd_value = pymoo_GD(true_pf_data)(algorithm_data)
    igd_value = pymoo_IGD(true_pf_data)(algorithm_data)
    hv_value = HV(ref_point=ref_point)(algorithm_data)
    spacing_value = calculate_spacing(algorithm_data)
    spread_value = calculate_spread(true_pf_data, algorithm_data)
    return gd_value, igd_value, hv_value, spacing_value, spread_value


def multi_objective_rao_with_tracking(pop_size, num_variables, var_ranges, max_iterations, archive_size, objective_functions, true_pf, ref_point):
    population = initialize_population(pop_size, num_variables, var_ranges)
    archive = []
    
    
    metrics_history = {
        "fe": [], "GD": [], "IGD": [], "SP": [], "SD": [], "HV": []
    }
    
    for sol in population:
        evaluate_objectives(sol, objective_functions)
        archive = update_archive(archive, sol, archive_size)
        
    for iteration in range(max_iterations):
        new_population = []
        for i in range(pop_size):
            x1 = random.choice(archive)
            x2 = random.choice(population)
            xi = population[i]
            r1, r2 = random.random(), random.random()
            new_vars = [max(0.0, min(1.0, xi.variables[j] + r1 * (x1.variables[j] - xi.variables[j]) + r2 * (xi.variables[j] - x2.variables[j])))
                        for j in range(num_variables)]
            new_sol = Solution(new_vars)
            evaluate_objectives(new_sol, objective_functions)
            archive = update_archive(archive, new_sol, archive_size)
            new_population.append(new_sol)
        population = new_population
        
        
        front_values = np.array([sol.objective_values for sol in archive])
        gd, igd, hv, spacing, spread_value = calculate_metrics(true_pf, front_values, ref_point)
        
        metrics_history["fe"].append((iteration + 1) * pop_size)
        metrics_history["GD"].append(gd)
        metrics_history["IGD"].append(igd)
        metrics_history["SP"].append(spacing)
        metrics_history["SD"].append(spread_value)
        metrics_history["HV"].append(hv)
        
    return archive, metrics_history




def plot_combined_metrics(history):
    """Combined metrics plot with updated external legend style for DTLZ4"""
    plt.figure(figsize=(10.5, 6))
    
    plt.plot(history["fe"], history["GD"], 'b-', marker='o', label="GD", markersize=2, linewidth=1.2, zorder=5)
    plt.plot(history["fe"], history["IGD"], 'g--', marker='d', label="IGD", markersize=2, linewidth=1.2, zorder=4)
    plt.plot(history["fe"], history["SP"], 'r:', marker='^', label="Spacing (SP)", markersize=2, linewidth=1.2, zorder=3)
    plt.plot(history["fe"], history["SD"], 'purple', linestyle='-.', marker='s', label="Spread (SD)", markersize=2, linewidth=1.2, zorder=2)
    plt.plot(history["fe"], history["HV"], color='orange', linestyle='--', marker='x', label="Hypervolume (HV)", markersize=2.5, linewidth=1.2, zorder=1)
    
    plt.title('Convergence of Performance Metrics for DTLZ4 Problem', fontsize=16, fontweight='bold', pad=12)
    plt.xlabel('Number of Function Evaluations', fontsize=15, labelpad=8)
    plt.ylabel('Metric Value', fontsize=15, labelpad=8)
    
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=12, frameon=True, facecolor='white', edgecolor='gray')
    
    plt.grid(False)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    
    plt.tight_layout()
    plt.savefig('dtlz4convg.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print("Saved: dtlz4convg.png successfully.")


# --- Main Parameters ---
num_variables = 12
variable_ranges = [(0, 1)] * num_variables
population_size = 100
max_iterations = 400
archive_size = 100
reference_point = [1, 1, 1]
objective_functions = [f1, f2, f3]
num_runs = 30
ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)
true_pf = get_problem("dtlz4").pareto_front(ref_dirs)


metrics = {"GD": [], "GD+": [], "IGD": [], "IGD+": [], "HV": [], "Spacing": [], "Coverage": [], "Spread": [], "Run Time": []}
all_fronts = []

start_time_total = time.time()

for run in range(num_runs):
    start_time_run = time.time()
    pareto_front, history = multi_objective_rao_with_tracking(
        population_size, num_variables, variable_ranges, max_iterations, archive_size, objective_functions, true_pf, reference_point
    )
    end_time_run = time.time()
    run_time = end_time_run - start_time_run
    front_values = np.array([sol.objective_values for sol in pareto_front])
    all_fronts.append(front_values)
    
    
    gd_val = pymoo_GD(true_pf)(front_values)
    gd_plus_val = GDPlus(true_pf)(front_values)
    igd_val = pymoo_IGD(true_pf)(front_values)
    igd_plus_val = IGDPlus(true_pf)(front_values)
    hv_val = HV(ref_point=reference_point)(front_values)
    spacing_val = calculate_spacing(front_values)
    coverage_val = calculate_coverage(true_pf, front_values)
    spread_val = calculate_spread(true_pf, front_values)
    
    metrics["GD"].append(gd_val)
    metrics["GD+"].append(gd_plus_val)
    metrics["IGD"].append(igd_val)
    metrics["IGD+"].append(igd_plus_val)
    metrics["HV"].append(hv_val)
    metrics["Spacing"].append(spacing_val)
    metrics["Coverage"].append(coverage_val)
    metrics["Spread"].append(spread_val)
    metrics["Run Time"].append(run_time)

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] Run {run + 1}: GD = {gd_val:.5f}, GD+ = {gd_plus_val:.5f}, IGD = {igd_val:.5f}, IGD+ = {igd_plus_val:.5f}, HV = {hv_val:.5f}, Spacing = {spacing_val:.5f}, Coverage = {coverage_val:.5f}, Spread = {spread_val:.5f}, Run Time = {run_time:.2f} seconds")


plot_combined_metrics(history)


final_front = all_fronts[-1]
fig = plt.figure(figsize=(8, 6.5))
ax = fig.add_subplot(111, projection='3d')


ax.scatter(true_pf[:, 0], true_pf[:, 1], true_pf[:, 2], 
           color='blue', s=30, alpha=0.6, marker='o', label='True Pareto Front', zorder=1)


ax.scatter(final_front[:, 0], final_front[:, 1], final_front[:, 2], 
           color='red', s=30, alpha=0.8, marker='o', label='Obtained Pareto Front', zorder=2)

plt.title('True vs. Obtained Pareto Front for DTLZ4', fontsize=16, fontweight='bold', pad=12)

ax.set_xlabel('$f_1$', fontsize=16, labelpad=5)
ax.set_ylabel('$f_2$', fontsize=16, labelpad=5)
ax.set_zlabel('$f_3$', fontsize=16, labelpad=-1)


ax.view_init(elev=30, azim=-135)
ax.invert_yaxis()
ax.invert_xaxis()

ax.legend(loc='upper right', fontsize=14, frameon=True, facecolor='white', framealpha=0.9)
ax.tick_params(axis='both', which='major', labelsize=13)
ax.grid(False)

plt.tight_layout()
plt.savefig('dtlz4paretofront.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

print("Saved: dtlz4paretofront.png successfully.")


print("\n--- Performance Metric Statistics ---")
for metric_name, values in metrics.items():
    valid_values = [v for v in values if not np.isnan(v)]
    if valid_values:
        best = np.min(valid_values) if metric_name not in ["HV", "Spread"] else np.max(valid_values)
        mean = np.mean(valid_values)
        std_dev = np.std(valid_values)
        print(f"{metric_name}: Best = {best:.5f}, Mean = {mean:.5f}, Std Dev = {std_dev:.5f}")
    else:
        print(f"{metric_name}: No valid values calculated.")