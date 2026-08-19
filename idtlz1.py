# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 20:59:04 2026

@author: Ajinkya Salve
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 16:52:51 2025

@author: Ajinkya Salve
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from pymoo.indicators.hv import HV
from pymoo.indicators.gd_plus import GDPlus
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions
from mpl_toolkits.mplot3d import Axes3D
import time
from datetime import datetime


def g(x):
    return (100 * (10 + sum((xi - 0.5) ** 2 - np.cos(20 * np.pi * (xi - 0.5)) for xi in x[2:])))

def f1(x):
    return 1 - (0.5 * x[0] * x[1] * (1 + g(x)))

def f2(x):
    return 1 - (0.5 * x[0] * (1 - x[1]) * (1 + g(x)))

def f3(x):
    return 1 - (0.5 * (1 - x[0]) * (1 + g(x)))


# === Solution Representation ===
class Solution:
    def __init__(self, variables):
        self.variables = variables
        self.objective_values = None
        self.crowding_distance = 0.0

# === Population Initialization and Evaluation ===
def initialize_population(pop_size, num_variables, var_ranges):
    return [Solution([random.uniform(low, high) for low, high in var_ranges]) for _ in range(pop_size)]

def evaluate_objectives(solution, objective_functions):
    solution.objective_values = [func(solution.variables) for func in objective_functions]

# === Pareto Dominance and Archive Update ===
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

# === Metric Calculations ===
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
    spread = calculate_spread(true_pf_data, algorithm_data)
    return gd, gd_plus, igd, igd_plus, hv, spacing, spread

# === Convergence Tracking (Target Code Format Verification) ===
convergence_data = {"fe": [], "GD": [], "IGD": [], "SP": [], "SD": [], "HV": []}

def multi_objective_rao_with_convergence(pop_size, num_variables, var_ranges, max_iter, archive_size, objective_functions,
                                         true_pf, ref_point):
    population = initialize_population(pop_size, num_variables, var_ranges)
    archive = []
    for sol in population:
        evaluate_objectives(sol, objective_functions)
        archive = update_archive(archive, sol, archive_size)

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

        # === Collect metrics per iteration ===
        front_values = [sol.objective_values for sol in archive]
        normalized_front = normalize_objectives(front_values)
        gd, _, igd, _, hv, spacing, spread = calculate_metrics(true_pf, normalized_front, ref_point)

        evaluations = it * pop_size
        convergence_data["fe"].append(evaluations)
        convergence_data["GD"].append(gd)
        convergence_data["IGD"].append(igd)
        convergence_data["HV"].append(hv)
        convergence_data["SP"].append(spacing)
        convergence_data["SD"].append(spread)

    return archive

# === Experiment Parameters ===
num_variables = 7
variable_ranges = [(0, 1)] * num_variables
population_size = 100
max_iterations = 100
archive_size = 100                        
reference_point = [1, 1, 1]
objective_functions = [f1, f2, f3]

ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)
true_pf = 1 - get_problem("dtlz1").pareto_front(ref_dirs)

# === Run the Algorithm with Convergence Tracking ===
pareto_front = multi_objective_rao_with_convergence(population_size, num_variables, variable_ranges,
                                                    max_iterations, archive_size, objective_functions,
                                                    true_pf, reference_point)

# === Final Convergence Plot (Matching Your Target Reference Style with 0.2 Y-Interval) ===
def plot_combined_metrics(history):
    """Combined metrics plot with updated external legend style"""
    plt.figure(figsize=(10.5, 6))
    
    plt.plot(history["fe"], history["GD"], 'b-', marker='o', label="GD", markersize=2, linewidth=1.2, zorder=5)
    plt.plot(history["fe"], history["IGD"], 'g--', marker='d', label="IGD", markersize=2, linewidth=1.2, zorder=4)
    plt.plot(history["fe"], history["SP"], 'r:', marker='^', label="Spacing (SP)", markersize=2, linewidth=1.2, zorder=3)
    plt.plot(history["fe"], history["SD"], 'purple', linestyle='-.', marker='s', label="Spread (SD)", markersize=2, linewidth=1.2, zorder=2)
    plt.plot(history["fe"], history["HV"], color='orange', linestyle='--', marker='x', label="Hypervolume (HV)", markersize=2.5, linewidth=1.2, zorder=1)
    
    plt.title('Convergence of Performance Metrics for IDTLZ1 Problem', fontsize=16, fontweight='bold', pad=12)
    plt.xlabel('Number of Function Evaluations', fontsize=15, labelpad=8)
    plt.ylabel('Metric Value', fontsize=15, labelpad=8)
    
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=12, frameon=True, facecolor='white', edgecolor='gray')
    
    plt.grid(False)
    plt.xticks(fontsize=13)
    
    # Calculate global max to maintain exact step size generation
    y_max = max(max(history["GD"]), max(history["IGD"]), max(history["SP"]), max(history["SD"]), max(history["HV"]), 1.0)
    plt.yticks(np.arange(0.0, y_max + 0.1, 0.2), fontsize=13)
    
    plt.tight_layout()
    plt.savefig('idtlz1convg.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print("Saved: idtlz1convg.png successfully with 0.2 Y-axis intervals.")

# === Call the Plot Function ===
plot_combined_metrics(convergence_data)