# -*- coding: utf-8 -*-
"""


@author: Ajinkya Salve
"""

#dtlz1
import random
import numpy as np
import matplotlib.pyplot as plt
from pymoo.indicators.hv import HV
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions
from datetime import datetime
import time
from mpl_toolkits.mplot3d import Axes3D


def g(x):
    """DTLZ1 auxiliary function"""
    return 100 * (10 + sum((xi - 0.5)**2 - np.cos(20 * np.pi * (xi - 0.5)) for xi in x[2:]))

def f1(x):
    """First objective function for DTLZ1"""
    return 0.5 * x[0] * x[1] * (1 + g(x))

def f2(x):
    """Second objective function for DTLZ1"""
    return 0.5 * x[0] * (1 - x[1]) * (1 + g(x))

def f3(x):
    """Third objective function for DTLZ1"""
    return 0.5 * (1 - x[0]) * (1 + g(x))


class Solution:
    """Represents a solution with variables and objective values"""
    def __init__(self, variables):
        self.variables = variables
        self.objective_values = None
        self.crowding_distance = 0.0


def initialize_population(pop_size, num_variables, var_ranges):
    """Initialize population with random solutions"""
    return [Solution([random.uniform(low, high) for low, high in var_ranges]) for _ in range(pop_size)]

def evaluate_objectives(solution, objective_functions):
    """Evaluate objective functions for a solution"""
    solution.objective_values = [func(solution.variables) for func in objective_functions]


def dominates(obj1, obj2):
    """Check if obj1 dominates obj2 (Pareto dominance)"""
    return all(a <= b for a, b in zip(obj1, obj2)) and any(a < b for a, b in zip(obj1, obj2))

def update_archive(archive, solution, archive_size):
    """Update archive with new solution, maintaining non-dominated solutions"""
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
    """Calculate crowding distance for diversity preservation"""
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


def normalize_objectives(objectives):
    """Normalize objectives by sum (DTLZ1 specific normalization)"""
    objectives = np.array(objectives)
    summed = np.sum(objectives, axis=1).reshape(-1, 1)
    return objectives / summed * 0.5


def gd_custom(PF, truePF, q=2):
    """Generational Distance (GD) metric"""
    PF = np.array(PF)
    truePF = np.array(truePF)
    m1 = PF.shape[0]
    
    max_vals = np.max(truePF, axis=0)
    min_vals = np.min(truePF, axis=0)
    normalized_PF = (PF - min_vals) / (max_vals - min_vals)
    normalized_true_PF = (truePF - min_vals) / (max_vals - min_vals)
    
    GD = 0.0
    for i in range(m1):
        diff = normalized_PF[i] - normalized_true_PF
        dist = np.sqrt(np.sum(diff**2, axis=1))
        GD += np.min(dist) ** q
    
    GD = (GD ** (1.0 / q)) / m1
    return GD

def igd_custom(PF, truePF, q=2):
    """Inverted Generational Distance (IGD) metric"""
    PF = np.array(PF)
    truePF = np.array(truePF)
    m = truePF.shape[0]
    
    max_vals = np.max(truePF, axis=0)
    min_vals = np.min(truePF, axis=0)
    normalized_PF = (PF - min_vals) / (max_vals - min_vals)
    normalized_true_PF = (truePF - min_vals) / (max_vals - min_vals)
    
    IGD = 0.0
    for i in range(m):
        diff = normalized_true_PF[i] - normalized_PF
        dist = np.sqrt(np.sum(diff**2, axis=1))
        IGD += np.min(dist) ** q
    
    IGD = (IGD ** (1.0 / q)) / m
    return IGD

def calculate_spacing(pareto_front):
    """Calculate spacing metric for distribution uniformity"""
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
    """Calculate spread metric for extent coverage"""
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

def calculate_metrics(true_pf_data, algorithm_data, ref_point):
    """Calculate all performance metrics"""
    gd_value, igd_value, hv_value, spacing_value, spread_value = np.nan, np.nan, np.nan, np.nan, np.nan
    
    if len(algorithm_data) > 0 and len(true_pf_data) > 0:
        gd_value = gd_custom(algorithm_data, true_pf_data)
        igd_value = igd_custom(algorithm_data, true_pf_data)
        
        ind_hv = HV(ref_point=ref_point)
        hv_value = ind_hv(algorithm_data)
        
        spacing_value = calculate_spacing(algorithm_data)
        spread_value = calculate_spread(true_pf_data, algorithm_data)
    elif len(algorithm_data) > 0:
        spacing_value = calculate_spacing(algorithm_data)
    
    return gd_value, igd_value, hv_value, spacing_value, spread_value


def multi_objective_rao_with_tracking(pop_size, num_variables, var_ranges, max_iter, archive_size, 
                                     objective_functions, true_pf, ref_point):
    """Multi-objective Rao algorithm with performance tracking"""
    population = initialize_population(pop_size, num_variables, var_ranges)
    archive = []
    
    metrics_history = {
        "fe": [], "GD": [], "IGD": [], "SP": [], "SD": [], "HV": []
    }

    for sol in population:
        evaluate_objectives(sol, objective_functions)
        archive = update_archive(archive, sol, archive_size)

    for iteration in range(max_iter):
        new_population = []
        for i in range(pop_size):
            x1 = random.choice(archive)
            x2 = random.choice(population)
            xi = population[i]
            
            r1, r2 = random.random(), random.random()
            
            new_vars = []
            for j in range(num_variables):
                new_val = xi.variables[j] + r1 * (x1.variables[j] - xi.variables[j]) + \
                          r2 * (xi.variables[j] - x2.variables[j])
                new_val = max(0.0, min(1.0, new_val))
                new_vars.append(new_val)
            
            new_sol = Solution(new_vars)
            evaluate_objectives(new_sol, objective_functions)
            
            archive = update_archive(archive, new_sol, archive_size)
            new_population.append(new_sol)
        
        population = new_population

        front_values = [sol.objective_values for sol in archive]
        normalized_front = normalize_objectives(front_values)
        
        gd, igd, hv, spacing, spread = calculate_metrics(true_pf, normalized_front, ref_point)

        metrics_history["fe"].append((iteration + 1) * pop_size)
        metrics_history["GD"].append(gd)
        metrics_history["IGD"].append(igd)
        metrics_history["SP"].append(spacing)
        metrics_history["SD"].append(spread)
        metrics_history["HV"].append(hv)

    return archive, metrics_history




def plot_combined_metrics(all_histories):
    """Combined metrics plot using the mean across all independent runs, with updated external legend style for DTLZ1"""
    plt.figure(figsize=(10.5, 6))

    function_evals = all_histories[0]["fe"]
    num_iterations = len(function_evals)

    # Average each metric, iteration-by-iteration, across all runs
    mean_history = {"GD": [], "IGD": [], "SP": [], "SD": [], "HV": []}
    for i in range(num_iterations):
        for metric in mean_history.keys():
            values_at_i = [history[metric][i] for history in all_histories if not np.isnan(history[metric][i])]
            mean_history[metric].append(np.mean(values_at_i) if values_at_i else np.nan)

    plt.plot(function_evals, mean_history["GD"], 'b-', marker='o', label="GD", markersize=2, linewidth=1.2, zorder=5)
    plt.plot(function_evals, mean_history["IGD"], 'g--', marker='d', label="IGD", markersize=2, linewidth=1.2, zorder=4)
    plt.plot(function_evals, mean_history["SP"], 'r:', marker='^', label="Spacing (SP)", markersize=2, linewidth=1.2, zorder=3)
    plt.plot(function_evals, mean_history["SD"], 'purple', linestyle='-.', marker='s', label="Spread (SD)", markersize=2, linewidth=1.2, zorder=2)
    plt.plot(function_evals, mean_history["HV"], color='orange', linestyle='--', marker='x', label="Hypervolume (HV)", markersize=2.5, linewidth=1.2, zorder=1)
    
    plt.title('Convergence of Performance Metrics for DTLZ1 Problem', fontsize=16, fontweight='bold', pad=12)
    plt.xlabel('Number of Function Evaluations', fontsize=15, labelpad=8)
    plt.ylabel('Metric Value', fontsize=15, labelpad=8)
    
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=12, frameon=True, facecolor='white', edgecolor='gray')
    
    plt.grid(False)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    
    plt.tight_layout()
    plt.savefig('dtlz1convg.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print("Saved: dtlz1convg.png successfully.")

    return mean_history


# --- Main Execution ---
if __name__ == "__main__":
    print("Starting Multi-Objective Proposed Algorithm for DTLZ1...")
    
    num_variables = 7
    variable_ranges = [(0, 1)] * num_variables
    population_size = 100
    max_iterations = 400
    archive_size = 100
    reference_point = [1, 1, 1]
    objective_functions = [f1, f2, f3]
    num_runs = 30

    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)
    true_pf = get_problem("dtlz1").pareto_front(ref_dirs)

    
    all_histories = []
    final_run_metrics_values = {"GD": [], "IGD": [], "HV": [], "Spacing": [], "Spread": []}
    run_times = []
    last_pareto_front = None
    start_time_total = time.time()

    for run in range(num_runs):
        start_time_run = time.time()
        pareto_front, history = multi_objective_rao_with_tracking(
            population_size, num_variables, variable_ranges,
            max_iterations, archive_size, objective_functions,
            true_pf, reference_point
        )
        end_time_run = time.time()
        run_time = end_time_run - start_time_run
        run_times.append(run_time)
        all_histories.append(history)
        last_pareto_front = pareto_front

        
        gd, igd, hv, spacing, spread = (
            history["GD"][-1], history["IGD"][-1], history["HV"][-1], history["SP"][-1], history["SD"][-1]
        )
        final_run_metrics_values["GD"].append(gd)
        final_run_metrics_values["IGD"].append(igd)
        final_run_metrics_values["HV"].append(hv)
        final_run_metrics_values["Spacing"].append(spacing)
        final_run_metrics_values["Spread"].append(spread)

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{current_time}] Run {run + 1}/{num_runs} completed in {run_time:.4f} seconds. "
              f"GD={gd:.6f}, IGD={igd:.6f}, HV={hv:.6f}, SP={spacing:.6f}, SD={spread:.6f}")

    end_time_total = time.time()
    total_time = end_time_total - start_time_total
    print(f"\nTotal Execution Time for {num_runs} runs: {total_time:.2f} seconds")

    
    mean_history = plot_combined_metrics(all_histories)

    
    final_front = normalize_objectives([sol.objective_values for sol in last_pareto_front])
    
    fig = plt.figure(figsize=(8, 6.5))
    ax = fig.add_subplot(111, projection='3d')
    
    
    ax.scatter(true_pf[:, 0], true_pf[:, 1], true_pf[:, 2], 
               color='blue', s=30, alpha=0.6, marker='o', label='True Pareto Front', zorder=1)
    
    
    ax.scatter(final_front[:, 0], final_front[:, 1], final_front[:, 2], 
               color='red', s=30, alpha=0.8, marker='o', label='Obtained Pareto Front', zorder=2)
    
    plt.title('True vs. Obtained Pareto Front for DTLZ1', fontsize=16, fontweight='bold', pad=12)
    
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
    plt.savefig('dtlz1paretofront.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print("Saved: dtlz1paretofront.png successfully.")

    
    print("\n--- Performance Metric Statistics (Final Archives, 30 Independent Runs) ---")
    mean_final_metrics = {}
    for metric_name, values in final_run_metrics_values.items():
        valid_values = [v for v in values if not np.isnan(v)]
        if valid_values:
            best = np.max(valid_values) if metric_name == "HV" else np.min(valid_values)
            mean = np.mean(valid_values)
            std_dev = np.std(valid_values)
            mean_final_metrics[metric_name] = mean
            print(f"{metric_name}: Best = {best:.8f}, Mean = {mean:.8f}, Std Dev = {std_dev:.8f}")
        else:
            print(f"{metric_name}: No valid values calculated.")

    avg_run_time = np.mean(run_times) if run_times else np.nan
    print(f"Average Run Time per run: {avg_run_time:.4f} seconds")

    
    print("\n--- Numerical Comparison: Plot End Average vs. Final Statistics Mean (for Verification) ---")
    for metric_name in ["GD", "IGD", "Spacing", "Spread", "HV"]:
        history_key = {"GD": "GD", "IGD": "IGD", "Spacing": "SP", "Spread": "SD", "HV": "HV"}[metric_name]
        plot_end_avg = mean_history[history_key][-1] if mean_history[history_key] else np.nan
        stat_table_mean = mean_final_metrics.get(metric_name, np.nan)

        print(f"{metric_name}: Plot End Average = {plot_end_avg:.8f}, Statistics Mean = {stat_table_mean:.8f}")

        if not np.isnan(plot_end_avg) and not np.isnan(stat_table_mean):
            if abs(plot_end_avg - stat_table_mean) > 1e-7:
                print(f"  --> Potential DISCREPANCY DETECTED for {metric_name}!")

    print("\nVisualization complete!")
