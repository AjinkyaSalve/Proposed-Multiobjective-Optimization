# -*- coding: utf-8 -*-
"""


@author: Ajinkya Salve
"""

#sdtlz2
import random
import numpy as np
import matplotlib.pyplot as plt
from pymoo.indicators.hv import HV
from pymoo.indicators.gd_plus import GDPlus
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions
from datetime import datetime
import time
from mpl_toolkits.mplot3d import Axes3D


def clip_variables(variables, lower=0.0, upper=1.0):
    return [max(lower, min(upper, v)) for v in variables]


def g(x):
    return sum((xi - 0.5) ** 2 for xi in x[2:])


def f1(x):
    return 1 * ((1 + g(x)) * np.cos(x[0] * np.pi / 2) * np.cos(x[1] * np.pi / 2))


def f2(x):
    return 10 * ((1 + g(x)) * np.cos(x[0] * np.pi / 2) * np.sin(x[1] * np.pi / 2))


def f3(x):
    return 100 * ((1 + g(x)) * np.sin(x[0] * np.pi / 2))



class Solution:
    def __init__(self, variables):
        self.variables = variables
        self.objective_values = None
        self.crowding_distance = 0.0


# --- Initialization ---
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
                    archive[i + 1].objective_values[m] - archive[i - 1].objective_values[m]) / (f_max - f_min + 1e-9)


def multi_objective_rao(pop_size, num_variables, var_ranges, max_iter, archive_size, objective_functions,
                        scaled_pf, reference_point, track_convergence=False):
    population = initialize_population(pop_size, num_variables, var_ranges)
    archive = []

    
    convergence_data = {
        "function_evals": [],
        "GD": [], "IGD": [], "HV": [], "Spacing": [], "Spread": []
    }

    for sol in population:
        evaluate_objectives(sol, objective_functions)
        archive = update_archive(archive, sol, archive_size)

    
    if track_convergence:
        front_values = [sol.objective_values for sol in archive]
        scaled_front = scale_to_expected_ranges(front_values)
        gd, _, igd, _, hv, spacing, _, spread = calculate_metrics(scaled_pf, scaled_front, reference_point)
        convergence_data["function_evals"].append(pop_size)
        convergence_data["GD"].append(gd)
        convergence_data["IGD"].append(igd)
        convergence_data["HV"].append(hv)
        convergence_data["Spacing"].append(spacing)
        convergence_data["Spread"].append(spread)

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
            archive = update_archive(archive, new_sol, archive_size)
            new_population.append(new_sol)
        population = new_population

        
        if track_convergence:
            front_values = [sol.objective_values for sol in archive]
            scaled_front = scale_to_expected_ranges(front_values)
            gd, _, igd, _, hv, spacing, _, spread = calculate_metrics(scaled_pf, scaled_front, reference_point)
            convergence_data["function_evals"].append(pop_size * (iteration + 2))
            convergence_data["GD"].append(gd)
            convergence_data["IGD"].append(igd)
            convergence_data["HV"].append(hv)
            convergence_data["Spacing"].append(spacing)
            convergence_data["Spread"].append(spread)

    return archive, convergence_data


def scale_to_expected_ranges(front):
    return np.array(front)


def normalize_for_hv(algorithm_data, true_pf_data):
    algo_array = np.array(algorithm_data)
    true_pf_array = np.array(true_pf_data)
    min_vals = true_pf_array.min(axis=0)
    max_vals = true_pf_array.max(axis=0)
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1.0
    return (algo_array - min_vals) / ranges


def gd_custom(PF, truePF, q=2):
    PF = np.array(PF)
    truePF = np.array(truePF)
    m1 = PF.shape[0]
    max_vals = np.max(truePF, axis=0)
    min_vals = np.min(truePF, axis=0)
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1.0
    normalized_PF = (PF - min_vals) / ranges
    normalized_true_PF = (truePF - min_vals) / ranges
    GD = 0.0
    for i in range(m1):
        diff = normalized_PF[i] - normalized_true_PF
        dist = np.sqrt(np.sum(diff ** 2, axis=1))
        GD += np.min(dist) ** q
    GD = (GD ** (1.0 / q)) / m1
    return GD


def igd_custom(PF, truePF, q=2):
    PF = np.array(PF)
    truePF = np.array(truePF)
    m = truePF.shape[0]
    max_vals = np.max(truePF, axis=0)
    min_vals = np.min(truePF, axis=0)
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1.0
    normalized_PF = (PF - min_vals) / ranges
    normalized_true_PF = (truePF - min_vals) / ranges
    IGD = 0.0
    for i in range(m):
        diff = normalized_true_PF[i] - normalized_PF
        dist = np.sqrt(np.sum(diff ** 2, axis=1))
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
    distances = [np.linalg.norm(algorithm_data[i + 1] - algorithm_data[i]) for i in range(len(algorithm_data) - 1)]
    mean_d = np.mean(distances)
    spread = (df + dl + np.sum(np.abs(distances - mean_d))) / (df + dl + (len(algorithm_data) - 1) * mean_d)
    return spread


def calculate_coverage(true_pf, algorithm_data):
    if len(algorithm_data) == 0 or len(true_pf) == 0:
        return np.nan
    true_pf_norm = (true_pf - true_pf.min(axis=0)) / (true_pf.max(axis=0) - true_pf.min(axis=0) + 1e-9)
    algorithm_data_norm = (algorithm_data - algorithm_data.min(axis=0)) / (
                algorithm_data.max(axis=0) - algorithm_data.min(axis=0) + 1e-9)
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
        
        normalized_algo = normalize_for_hv(algorithm_data, true_pf_data)
        ind_hv = HV(ref_point=ref_point)
        
        gd_plus_value = ind_gd_plus(algorithm_data)
        igd_plus_value = ind_igd_plus(algorithm_data)
        hv_value = ind_hv(normalized_algo)
        spacing_value = calculate_spacing(algorithm_data)
        coverage_value = calculate_coverage(true_pf_data, algorithm_data)
        spread_value = calculate_spread(true_pf_data, algorithm_data)
    elif len(algorithm_data) > 0:
        spacing_value = calculate_spacing(algorithm_data)
    return gd_value, gd_plus_value, igd_value, igd_plus_value, hv_value, spacing_value, coverage_value, spread_value


# --- Main Execution
num_variables = 12
variable_ranges = [(0, 1)] * num_variables
population_size = 100
max_iterations = 100
archive_size = 100
reference_point = [1, 1, 1]
objective_functions = [f1, f2, f3]
num_runs = 30


ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)
pf = get_problem("dtlz2").pareto_front(ref_dirs)
scaling_factors = np.array([1, 10, 100])
scaled_pf = pf * scaling_factors


metrics = {"GD": [], "GD+": [], "IGD": [], "IGD+": [], "HV": [], "Spacing": [], "Coverage": [], "Spread": [],
           "Run Time": []}
all_fronts = []
all_convergence_data = []

start_time_total = time.time()

for run in range(num_runs):
    start_time = time.time()
    pareto_front, convergence_data = multi_objective_rao(
        population_size, num_variables, variable_ranges, max_iterations,
        archive_size, objective_functions, scaled_pf, reference_point,
        track_convergence=True
    )
    end_time = time.time()
    run_time = end_time - start_time

    front_values = [sol.objective_values for sol in pareto_front]
    scaled_front = scale_to_expected_ranges(front_values)
    all_fronts.append(scaled_front)
    all_convergence_data.append(convergence_data)

    gd, gd_plus, igd, igd_plus, hv, spacing, coverage, spread = calculate_metrics(scaled_pf, scaled_front,
                                                                                  reference_point)

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
    print(
        f"[{now}] Run {run + 1}: GD={gd:.8f}, GD+={gd_plus:.8f}, IGD={igd:.8f}, IGD+={igd_plus:.8f}, HV={hv:.8f}, "
        f"Spacing={spacing:.8f}, Coverage={coverage:.8f}, Spread={spread:.8f}, Run Time={run_time:.8f}s")



def plot_combined_metrics(all_convergence_data):
    """Combined metrics plot matching target reference style exactly with Normalized Spacing"""
    plt.figure(figsize=(10.5, 6))

    max_evals = max(len(run_data["function_evals"]) for run_data in all_convergence_data)
    function_evals = np.array(all_convergence_data[0]["function_evals"])

    
    mean_metrics = {m: [] for m in ["GD", "IGD", "Spacing", "Spread", "HV"]}
    for i in range(max_evals):
        for metric in mean_metrics.keys():
            values_at_i = [run_data[metric][i] for run_data in all_convergence_data
                           if i < len(run_data[metric]) and not np.isnan(run_data[metric][i])]
            mean_metrics[metric].append(np.mean(values_at_i) if values_at_i else np.nan)

    for m in mean_metrics:
        mean_metrics[m] = np.array(mean_metrics[m])

    
    sp_min = np.nanmin(mean_metrics["Spacing"])
    sp_max = np.nanmax(mean_metrics["Spacing"])
    if sp_max - sp_min > 0:
        normalized_spacing = (mean_metrics["Spacing"] - sp_min) / (sp_max - sp_min)
    else:
        normalized_spacing = mean_metrics["Spacing"]

    
    plt.plot(function_evals, mean_metrics["GD"], 'b-', marker='o', label="GD", markersize=2, linewidth=1.2, zorder=5)
    plt.plot(function_evals, mean_metrics["IGD"], 'g--', marker='d', label="IGD", markersize=2, linewidth=1.2, zorder=4)
    plt.plot(function_evals, normalized_spacing, 'r:', marker='^', label="Spacing (SP)", markersize=2, linewidth=1.2, zorder=3)
    plt.plot(function_evals, mean_metrics["Spread"], 'purple', linestyle='-.', marker='s', label="Spread (SD)", markersize=2, linewidth=1.2, zorder=2)
    plt.plot(function_evals, mean_metrics["HV"], color='orange', linestyle='--', marker='x', label="Hypervolume (HV)", markersize=2.5, linewidth=1.2, zorder=1)

    plt.title('Convergence of Performance Metrics for SDTLZ2 Problem', fontsize=16, fontweight='bold', pad=12)
    plt.xlabel('Number of Function Evaluations', fontsize=15, labelpad=8)
    plt.ylabel('Metric Value', fontsize=15, labelpad=8)

    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=12, frameon=True, facecolor='white', edgecolor='gray')

    
    plt.ylim(-0.05, 1.05)
    plt.grid(False)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    
    plt.tight_layout()
    plt.savefig('sdtlz2-convgernce.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print("Saved: sdtlz2-convgernce.png successfully with normalized visual fields.")


def plot_3d_pareto_front(final_front, true_pf):
    fig = plt.figure(figsize=(8, 6.5))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(true_pf[:, 0], true_pf[:, 1], true_pf[:, 2], 
               color='blue', s=30, alpha=0.6, label='True Pareto Front', zorder=1)
    ax.scatter(final_front[:, 0], final_front[:, 1], final_front[:, 2], 
               color='red', s=30, alpha=0.8, label='Obtained Pareto Front', zorder=2)

    plt.title('True vs. Obtained Pareto Front for SDTLZ2', fontsize=16, fontweight='bold', pad=12)

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
    plt.savefig('sdtlz2paretofront.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print("Saved: sdtlz2-paretofront.png successfully.")



plot_combined_metrics(all_convergence_data)

final_front = all_fronts[-1]
plot_3d_pareto_front(final_front, scaled_pf)


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