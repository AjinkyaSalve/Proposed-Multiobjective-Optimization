# -*- coding: utf-8 -*-
"""


@author: Ajinkya Salve
"""

#dtlz5
import random
import numpy as np
import matplotlib.pyplot as plt
from pymoo.indicators.hv import HV
from pymoo.indicators.gd_plus import GDPlus
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
import time


def g(x):
    x = np.array(x)
    return np.sum((x[3:] - 0.5)**2)

def theta1(x):
    return x[0] * np.pi / 2

def theta2(x):
    return (np.pi / (4 * (1 + g(x)))) * (1 + 2 * x[1] * g(x))

def f1(x):
    return (1 + g(x)) * np.cos(theta1(x)) * np.cos(theta2(x))

def f2(x):
    return (1 + g(x)) * np.cos(theta1(x)) * np.sin(theta2(x))

def f3(x):
    return (1 + g(x)) * np.sin(theta1(x))


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


def normalize_objectives(front):
    front = np.array(front)
    f_min = front.min(axis=0)
    f_max = front.max(axis=0)
    return (front - f_min) / (f_max - f_min + 1e-9)

def gd_custom(PF, truePF, q=2):
    PF, truePF = np.array(PF), np.array(truePF)
    max_vals, min_vals = np.max(truePF, axis=0), np.min(truePF, axis=0)
    PF = (PF - min_vals) / (max_vals - min_vals + 1e-9)
    truePF = (truePF - min_vals) / (max_vals - min_vals + 1e-9)
    GD = sum(np.min(np.linalg.norm(truePF - pf, axis=1)) ** q for pf in PF)
    return (GD / len(PF)) ** (1.0 / q)

def igd_custom(PF, truePF, q=2):
    PF, truePF = np.array(PF), np.array(truePF)
    max_vals, min_vals = np.max(truePF, axis=0), np.min(truePF, axis=0)
    PF = (PF - min_vals) / (max_vals - min_vals + 1e-9)
    truePF = (truePF - min_vals) / (max_vals - min_vals + 1e-9)
    IGD = sum(np.min(np.linalg.norm(PF - true, axis=1)) ** q for true in truePF)
    return (IGD / len(truePF)) ** (1.0 / q)

def calculate_spacing(pareto_front):
    N = len(pareto_front)
    if N <= 1: return np.nan
    distances = [min(np.linalg.norm(pf - other) for j, other in enumerate(pareto_front) if i != j)
                 for i, pf in enumerate(pareto_front)]
    mean_d = np.mean(distances)
    return np.sqrt(np.sum((distances - mean_d) ** 2) / (N - 1))

def calculate_spread(true_pf, algorithm_data):
    if len(algorithm_data) <= 1: return np.nan
    true_pf, algorithm_data = np.array(true_pf), np.array(algorithm_data)
    true_pf = true_pf[np.argsort(true_pf[:, 0])]
    algorithm_data = algorithm_data[np.argsort(algorithm_data[:, 0])]
    df = np.linalg.norm(algorithm_data[0] - true_pf[0])
    dl = np.linalg.norm(algorithm_data[-1] - true_pf[-1])
    distances = [np.linalg.norm(algorithm_data[i+1] - algorithm_data[i]) for i in range(len(algorithm_data)-1)]
    mean_d = np.mean(distances)
    return (df + dl + np.sum(np.abs(distances - mean_d))) / (df + dl + (len(algorithm_data) - 1) * mean_d)

def calculate_coverage(true_pf, algorithm_data):
    if len(algorithm_data) == 0 or len(true_pf) == 0: return np.nan
    true_pf_norm = normalize_objectives(true_pf)
    algorithm_data_norm = normalize_objectives(algorithm_data)
    distances = np.min(np.linalg.norm(algorithm_data_norm[:, None, :] - true_pf_norm[None, :, :], axis=2), axis=1)
    return np.mean(distances)

def calculate_metrics(true_pf_data, algorithm_data, ref_point):
    if len(algorithm_data) == 0 or len(true_pf_data) == 0:
        return (np.nan,) * 8
    algorithm_data = np.array(algorithm_data)
    gd = gd_custom(algorithm_data, true_pf_data)
    igd = igd_custom(algorithm_data, true_pf_data)
    gd_plus = GDPlus(true_pf_data)(algorithm_data)
    igd_plus = IGDPlus(true_pf_data)(algorithm_data)
    hv = HV(ref_point=ref_point)(normalize_objectives(algorithm_data))
    spacing = calculate_spacing(algorithm_data)
    coverage = calculate_coverage(true_pf_data, algorithm_data)
    spread = calculate_spread(true_pf_data, algorithm_data)
    return gd, gd_plus, igd, igd_plus, hv, spacing, coverage, spread


def multi_objective_rao(pop_size, num_variables, var_ranges, max_iter, archive_size, objective_functions,
                         track_convergence=False, true_pf=None, ref_point=None):
    population = initialize_population(pop_size, num_variables, var_ranges)
    archive = []
    for sol in population:
        evaluate_objectives(sol, objective_functions)
        archive = update_archive(archive, sol, archive_size)

    convergence_data = {"fe": [], "GD": [], "IGD": [], "SP": [], "SD": [], "HV": []} if track_convergence else None

    for it in range(1, max_iter + 1):
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
            normalized_front = normalize_objectives(front_values)
            gd, _, igd, _, hv, spacing, _, spread = calculate_metrics(true_pf, normalized_front, ref_point)

            evaluations = it * pop_size
            convergence_data["fe"].append(evaluations)
            convergence_data["GD"].append(gd)
            convergence_data["IGD"].append(igd)
            convergence_data["HV"].append(hv)
            convergence_data["SP"].append(spacing)
            convergence_data["SD"].append(spread)

    if track_convergence:
        return archive, convergence_data
    return archive


def plot_combined_metrics(history):
    """Combined metrics plot with external legend style"""
    plt.figure(figsize=(10.5, 6))
    
    plt.plot(history["fe"], history["GD"], 'b-', marker='o', label="GD", markersize=2, linewidth=1.2, zorder=5)
    plt.plot(history["fe"], history["IGD"], 'g--', marker='d', label="IGD", markersize=2, linewidth=1.2, zorder=4)
    plt.plot(history["fe"], history["SP"], 'r:', marker='^', label="Spacing (SP)", markersize=2, linewidth=1.2, zorder=3)
    plt.plot(history["fe"], history["SD"], 'purple', linestyle='-.', marker='s', label="Spread (SD)", markersize=2, linewidth=1.2, zorder=2)
    plt.plot(history["fe"], history["HV"], color='orange', linestyle='--', marker='x', label="Hypervolume (HV)", markersize=2.5, linewidth=1.2, zorder=1)
    
    plt.title('Convergence of Performance Metrics for DTLZ5 Problem', fontsize=16, fontweight='bold', pad=12)
    plt.xlabel('Number of Function Evaluations', fontsize=15, labelpad=8)
    plt.ylabel('Metric Value', fontsize=15, labelpad=8)
    
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=12, frameon=True, facecolor='white', edgecolor='gray')
    
    plt.grid(False)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    
    plt.tight_layout()
    plt.savefig('dtlz5convg.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print("Saved: dtlz5convg.png successfully.")

def plot_pareto_front(true_pf, final_front):
    """3D Pareto Plot formatted for DTLZ5 curve alignment"""
    fig = plt.figure(figsize=(8, 6.5))
    ax = fig.add_subplot(111, projection='3d')

    
    ax.scatter(true_pf[:, 0], true_pf[:, 1], true_pf[:, 2], 
               color='blue', s=30, alpha=0.6, marker='o', label='True Pareto Front', zorder=1)

    
    ax.scatter(final_front[:, 0], final_front[:, 1], final_front[:, 2], 
               color='red', s=30, alpha=0.8, marker='o', label='Obtained Pareto Front', zorder=2)

    plt.title('True vs. Obtained Pareto Front for DTLZ5', fontsize=16, fontweight='bold', pad=12)

    ax.set_xlabel('$f_1$', fontsize=16, labelpad=5)
    ax.set_ylabel('$f_2$', fontsize=16, labelpad=5)
    ax.set_zlabel('$f_3$', fontsize=16, labelpad=-1)

    ax.view_init(elev=20, azim=25)

    ax.legend(loc='upper right', fontsize=12, frameon=True, facecolor='white', framealpha=0.9)
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.grid(False)

    plt.tight_layout()
    plt.savefig('dtlz5paretofront.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print("Saved: dtlz5paretofront.png successfully.")


if __name__ == "__main__":
    
    num_variables = 12
    variable_ranges = [(0, 1)] * num_variables
    population_size = 100
    max_iterations = 400
    archive_size = 100                        
    reference_point = [1, 1, 1]
    objective_functions = [f1, f2, f3]
    num_runs = 30

    true_pf = get_problem("dtlz5").pareto_front()

    
    metrics = {"GD": [], "GD+": [], "IGD": [], "IGD+": [], "HV": [], "Spacing": [], "Coverage": [], "Spread": [], "Run Time": []}
    all_fronts = []
    last_convergence_data = None

    print("--- Starting 30-Run Optimization for DTLZ5 ---")
    for run in range(num_runs):
        start_time = time.time()
        
        
        if run == num_runs - 1:
            pareto_front, last_convergence_data = multi_objective_rao(
                population_size, num_variables, variable_ranges, max_iterations,
                archive_size, objective_functions, track_convergence=True,
                true_pf=true_pf, ref_point=reference_point
            )
        else:
            pareto_front = multi_objective_rao(
                population_size, num_variables, variable_ranges, max_iterations,
                archive_size, objective_functions
            )

        run_time = time.time() - start_time
        front_values = np.array([sol.objective_values for sol in pareto_front])
        all_fronts.append(front_values)

        normalized_front = normalize_objectives(front_values)
        gd, gd_plus, igd, igd_plus, hv, spacing, coverage, spread = calculate_metrics(true_pf, normalized_front, reference_point)
        
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
        print(f"[{now}] Run {run+1}: GD={gd:.8f}, GD+={gd_plus:.8f}, IGD={igd:.8f}, IGD+={igd_plus:.8f}, "
              f"HV={hv:.8f}, Spacing={spacing:.8f}, Coverage={coverage:.8f}, Spread={spread:.8f}, Time={run_time:.8f}s")

    
    plot_pareto_front(true_pf, all_fronts[-1])
    plot_combined_metrics(last_convergence_data)

    
    print("\n--- Performance Metric Statistics (30 Runs) ---")
    for metric, values in metrics.items():
        valid = [v for v in values if not np.isnan(v)]
        if valid:
            best = max(valid) if metric == "HV" else min(valid)
            mean = np.mean(valid)
            std = np.std(valid)
            print(f"{metric}: Best = {best:.8f}, Mean = {mean:.8f}, Std Dev = {std:.8f}")
        else:
            print(f"{metric}: No valid values.")