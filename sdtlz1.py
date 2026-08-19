# -*- coding: utf-8 -*-
"""


@author: Ajinkya Salve
"""

#sdtlz1
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

def clip_variables(variables, lower=0.0, upper=1.0):
    return [max(lower, min(upper, v)) for v in variables]


def g(x):
    return 100 * (10 + sum((xi - 0.5)**2 - np.cos(20 * np.pi * (xi - 0.5)) for xi in x[2:]))

def f1(x):
    return 0.5 * x[0] * x[1] * (1 + g(x))

def f2(x):
    return 10 * (0.5 * x[0] * (1 - x[1]) * (1 + g(x)))

def f3(x):
    return 100 * (0.5 * (1 - x[0]) * (1 + g(x)))


class Solution:
    def __init__(self, variables):
        self.variables = variables
        self.objective_values = None
        self.crowding_distance = 0.0


def initialize_population(pop_size, num_variables, var_ranges):
    return [Solution([random.uniform(low, high) for low, high in var_ranges]) for _ in range(pop_size)]

def evaluate_objectives(solution, objective_functions):
    solution.objective_values = [func(solution.variables) for func in objective_functions]


def dominates(obj1, obj2):
    return all(a <= b for a, b in zip(obj1, obj2)) and any(a < b for a, b in zip(obj1, obj2))

def update_archive(archive, solution, archive_size):
    new_archive = []
    non_dominated = True
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
            archive[i].crowding_distance += (
                archive[i + 1].objective_values[m] - archive[i - 1].objective_values[m]) / (f_max - f_min)


def multi_objective_rao_tracked(pop_size, num_variables, var_ranges, max_iter, archive_size, objective_functions, scaled_pf, ref_point=[1, 1, 1]):
    population = initialize_population(pop_size, num_variables, var_ranges)
    archive = []
    
    # Initialize metric history storage matching your format
    history = {"fe": [], "GD": [], "IGD": [], "SP": [], "SD": [], "HV": []}
    fe_counter = 0
    
    for sol in population:
        evaluate_objectives(sol, objective_functions)
        fe_counter += 1
        archive = update_archive(archive, sol, archive_size)
        
    for iteration in range(max_iter):
        new_population = []
        for i in range(pop_size):
            x1 = random.choice(archive)
            x2 = random.choice(population)
            xi = population[i]
            r1, r2 = random.random(), random.random()
            new_vars = [
                max(0.0, min(1.0, xi.variables[j] + r1 * (x1.variables[j] - xi.variables[j]) +
                              r2 * (xi.variables[j] - x2.variables[j])))
                for j in range(num_variables)
            ]
            new_sol = Solution(new_vars)
            evaluate_objectives(new_sol, objective_functions)
            fe_counter += 1
            archive = update_archive(archive, new_sol, archive_size)
            new_population.append(new_sol)
        population = new_population
        
        
        current_front = [sol.objective_values for sol in archive]
        scaled_current = scale_to_expected_ranges(current_front)
        
        gd_val, _, igd_val, _, hv_val, sp_val, _, sd_val = calculate_metrics(scaled_pf, scaled_current, ref_point)
        
        
        history["fe"].append(fe_counter)
        history["GD"].append(gd_val if not np.isnan(gd_val) else 0.0)
        history["IGD"].append(igd_val if not np.isnan(igd_val) else 0.0)
        history["SP"].append(sp_val if not np.isnan(sp_val) else 0.0)
        history["SD"].append(sd_val if not np.isnan(sd_val) else 0.0)
        history["HV"].append(hv_val if not np.isnan(hv_val) else 0.0)
        
    return archive, history

def scale_to_expected_ranges(front):
    front = np.array(front)
    if len(front) == 0:
        return np.zeros((0, 3))
    f1_scaled = front[:, 0] / 600   
    f2_scaled = front[:, 1] / 6000  
    f3_scaled = front[:, 2] / 60000 
    return np.stack([f1_scaled * 1, f2_scaled * 10, f3_scaled * 100], axis=1)

def gd_custom(PF, truePF, q=2):
    PF = np.array(PF)
    truePF = np.array(truePF)
    m1 = PF.shape[0]
    max_vals = np.max(truePF, axis=0)
    min_vals = np.min(truePF, axis=0)
    normalized_PF = (PF - min_vals) / (max_vals - min_vals + 1e-9)
    normalized_true_PF = (truePF - min_vals) / (max_vals - min_vals + 1e-9)
    GD = 0.0
    for i in range(m1):
        diff = normalized_PF[i] - normalized_true_PF
        dist = np.sqrt(np.sum(diff**2, axis=1))
        GD += np.min(dist) ** q
    return (GD ** (1.0 / q)) / m1

def igd_custom(PF, truePF, q=2):
    PF = np.array(PF)
    truePF = np.array(truePF)
    m = truePF.shape[0]
    max_vals = np.max(truePF, axis=0)
    min_vals = np.min(truePF, axis=0)
    normalized_PF = (PF - min_vals) / (max_vals - min_vals + 1e-9)
    normalized_true_PF = (truePF - min_vals) / (max_vals - min_vals + 1e-9)
    IGD = 0.0
    for i in range(m):
        diff = normalized_true_PF[i] - normalized_PF
        dist = np.sqrt(np.sum(diff**2, axis=1))
        IGD += np.min(dist) ** q
    return (IGD ** (1.0 / q)) / m

def calculate_spacing(pareto_front):
    N = len(pareto_front)
    if N <= 1: return np.nan
    distances = []
    for i in range(N):
        dists = [np.linalg.norm(pareto_front[i] - pareto_front[j]) for j in range(N) if i != j]
        distances.append(min(dists))
    distances = np.array(distances)
    mean_d = np.mean(distances)
    return np.sqrt(np.sum((distances - mean_d) ** 2) / (N - 1))

def calculate_spread(true_pf, algorithm_data):
    if len(algorithm_data) <= 1: return np.nan
    true_pf, algorithm_data = np.array(true_pf), np.array(algorithm_data)
    true_pf = true_pf[np.argsort(true_pf[:, 0])]
    algorithm_data = algorithm_data[np.argsort(algorithm_data[:, 0])]
    df = np.linalg.norm(algorithm_data[0] - true_pf[0])
    dl = np.linalg.norm(algorithm_data[-1] - true_pf[-1])
    distances = [np.linalg.norm(algorithm_data[i+1] - algorithm_data[i]) for i in range(len(algorithm_data) - 1)]
    mean_d = np.mean(distances)
    return (df + dl + np.sum(np.abs(distances - mean_d))) / (df + dl + (len(algorithm_data) - 1) * mean_d)

def calculate_coverage(true_pf, algorithm_data):
    if len(algorithm_data) == 0 or len(true_pf) == 0: return np.nan
    true_pf_norm = (true_pf - true_pf.min(axis=0)) / (true_pf.max(axis=0) - true_pf.min(axis=0) + 1e-9)
    algorithm_data_norm = (algorithm_data - algorithm_data.min(axis=0)) / (algorithm_data.max(axis=0) - algorithm_data.min(axis=0) + 1e-9)
    min_distances = np.min(np.linalg.norm(algorithm_data_norm[:, None, :] - true_pf_norm[None, :, :], axis=-1), axis=1)
    return np.mean(min_distances)

def calculate_metrics(true_pf_data, algorithm_data, ref_point):
    gd_value, gd_plus_value, igd_value, igd_plus_value, hv_value, spacing_value, coverage_value, spread_value = [np.nan]*8
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


def plot_combined_metrics(history):
    """Combined metrics plot with updated external legend style"""
    plt.figure(figsize=(10.5, 6))
    
    plt.plot(history["fe"], history["GD"], 'b-', marker='o', label="GD", markersize=2, linewidth=1.2, zorder=5)
    plt.plot(history["fe"], history["IGD"], 'g--', marker='d', label="IGD", markersize=2, linewidth=1.2, zorder=4)
    plt.plot(history["fe"], history["SP"], 'r:', marker='^', label="Spacing (SP)", markersize=2, linewidth=1.2, zorder=3)
    plt.plot(history["fe"], history["SD"], 'purple', linestyle='-.', marker='s', label="Spread (SD)", markersize=2, linewidth=1.2, zorder=2)
    plt.plot(history["fe"], history["HV"], color='orange', linestyle='--', marker='x', label="Hypervolume (HV)", markersize=2.5, linewidth=1.2, zorder=1)
    
    plt.title('Convergence of Performance Metrics for SDTLZ1 Problem', fontsize=16, fontweight='bold', pad=12)
    plt.xlabel('Number of Function Evaluations', fontsize=15, labelpad=8)
    plt.ylabel('Metric Value', fontsize=15, labelpad=8)
    
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=12, frameon=True, facecolor='white', edgecolor='gray')
    
    plt.grid(False)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    
    plt.tight_layout()
    plt.savefig('sdtlz1convg.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print("Saved: sdtlz1convg.png successfully.")

# --- Main Execution ---
num_variables = 7
variable_ranges = [(0, 1)] * num_variables
population_size = 100
max_iterations  = 100
archive_size = 100
reference_point = [1, 1, 1]
objective_functions = [f1, f2, f3]
num_runs = 30


ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)
pf = get_problem("dtlz1").pareto_front(ref_dirs)
scaling_factors = np.array([1, 10, 100])
scaled_pf = pf * scaling_factors


metrics = {"GD": [], "GD+": [], "IGD": [], "IGD+": [], "HV": [], "Spacing": [], "Coverage": [], "Spread": [], "Run Time": []}
all_fronts = []
last_run_history = None

for run in range(num_runs):
    start_time = time.time()
    pareto_front, history = multi_objective_rao_tracked(population_size, num_variables, variable_ranges, max_iterations, archive_size, objective_functions, scaled_pf, reference_point)
    run_time = time.time() - start_time

    front_values = [sol.objective_values for sol in pareto_front]
    scaled_front = scale_to_expected_ranges(front_values)
    all_fronts.append(scaled_front)
    
    if run == num_runs - 1:
        last_run_history = history # Cache final execution trace history
    
    gd, gd_plus, igd, igd_plus, hv, spacing, coverage, spread = calculate_metrics(scaled_pf, scaled_front, reference_point)

    metrics["GD"].append(gd)
    metrics["GD+"].append(gd_plus)
    metrics["IGD"].append(igd)
    metrics["IGD+"].append(igd_plus)
    metrics["HV"].append(hv)
    metrics["Spacing"].append(spacing)
    metrics["Coverage"].append(coverage)
    metrics["Spread"].append(spread)
    metrics["Run Time"].append(run_time)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] Run {run+1}: GD={gd:.8f}, GD+={gd_plus:.8f}, IGD={igd:.8f}, IGD+={igd_plus:.8f}, HV={hv:.8f}, "
          f"Spacing={spacing:.8f}, Coverage={coverage:.8f}, Spread={spread:.8f}, Run Time={run_time:.8f}s")


final_front = all_fronts[-1]
fig = plt.figure(figsize=(8, 6.5))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(scaled_pf[:, 0], scaled_pf[:, 1], scaled_pf[:, 2], color='blue', s=30, alpha=0.6, marker='o', label='True Pareto Front', zorder=1)
ax.scatter(final_front[:, 0], final_front[:, 1], final_front[:, 2], color='red', s=30, alpha=0.8, marker='o', label='Obtained Pareto Front', zorder=2)

plt.title('True vs. Obtained Pareto Front for SDTLZ1', fontsize=16, fontweight='bold', pad=12)
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
plt.savefig('sdtlz1_pareto_front.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

if last_run_history is not None:
    plot_combined_metrics(last_run_history)


print("\n--- Performance Metric Statistics ---")
for metric, values in metrics.items():
    valid = [v for v in values if not np.isnan(v)]
    if valid:
        best = np.max(valid) if metric in ["HV"] else np.min(valid)
        mean = np.mean(valid)
        std = np.std(valid)
        print(f"{metric}: Best = {best:.8f}, Mean = {mean:.8f}, Std Dev = {std:.8f}")
    else:
        print(f"{metric}: No valid values.")