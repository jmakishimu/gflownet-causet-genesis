#
# gflownet-causet-genesis/analyze.py
#
import pandas as pd
import numpy as np
import pickle
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Import the reward proxy to re-compute metrics
from causet_reward import CausalSetRewardProxy

sns.set_theme(style="whitegrid")

def load_results(run_dir):
    """Loads log and ensemble from a run directory."""
    log_path = os.path.join(run_dir, "training_log.csv")
    ensemble_path = os.path.join(run_dir, "final_ensemble.pkl")

    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log file not found: {log_path}")
    if not os.path.exists(ensemble_path):
        raise FileNotFoundError(f"Ensemble file not found: {ensemble_path}")

    logs = pd.read_csv(log_path)

    with open(ensemble_path, 'rb') as f:
        ensemble = pickle.load(f)

    return logs, ensemble

def load_baseline(baseline_file):
    """Loads the random baseline ensemble."""
    if not os.path.exists(baseline_file):
        raise FileNotFoundError(f"Baseline file not found: {baseline_file}")

    with open(baseline_file, 'rb') as f:
        baseline_ensemble = pickle.load(f)
    return baseline_ensemble

def plot_1_loss(logs, save_path):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=logs, x='step', y='loss')
    plt.title("Plot 1: GFlowNet Loss Curve")
    plt.xlabel("Training Step")
    plt.ylabel("GFlowNet Loss (Trajectory Balance)")
    plt.yscale('log')
    plt.savefig(os.path.join(save_path, "1_loss_curve.png"))
    plt.close()

def plot_2_reward(logs, save_path):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=logs, x='step', y='avg_reward')
    plt.title("Plot 2: Reward Acquisition")
    plt.xlabel("Training Step")
    plt.ylabel("Average Reward R(c) of Batch")
    plt.savefig(os.path.join(save_path, "2_reward_acquisition.png"))
    plt.close()

def analyze_ensemble(ensemble, metric_to_calc):
    """Helper to calculate metrics for a full ensemble."""
    mmd_proxy = CausalSetRewardProxy(reward_type='mmd')
    bd_proxy = CausalSetRewardProxy(reward_type='bd')

    mmds = []
    bds = []

    if not ensemble:
        print("Warning: Ensemble is empty.")
        return {'mmds': [], 'bds': []}

    # (Fix #4 Applied) Filter malformed graphs based on N of the first graph,
    # as proxy does not have 'max_nodes' attribute.
    target_n = ensemble[0].number_of_nodes()
    valid_ensemble = [g for g in ensemble if g.number_of_nodes() == target_n]

    if len(valid_ensemble) < len(ensemble):
        print(f"Warning: Filtered {len(ensemble) - len(valid_ensemble)} malformed graphs.")

    desc = f"Analyzing {len(valid_ensemble)} graphs (N={target_n}) for {metric_to_calc}"
    for g in tqdm(valid_ensemble, desc=desc):
        if metric_to_calc == 'mmd':
            mmds.append(mmd_proxy.calculate_avg_mmd(g))
        elif metric_to_calc == 'bd':
            bds.append(bd_proxy.get_bd_energy(g))

    return {'mmds': mmds, 'bds': bds}

def plot_3_dim_locking(logs, ensemble, save_path):
    """
    This plot requires running the 'mmd' experiment.
    The 'avg_energy' column will be (MMD - 4.0)^2.
    """
    if 'avg_energy' not in logs.columns:
        print("Skipping Plot 3: 'avg_energy' not in logs.")
        return

    # Recalculate Avg MMD from energy
    logs['avg_mmd'] = logs['avg_energy'].apply(lambda e: np.sqrt(e) + 4.0)

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=logs, x='step', y='avg_mmd')
    plt.axhline(y=4.0, color='r', linestyle='--', label="Target d=4")
    plt.title("Plot 3: Dimension Locking (MMD Sanity Check)")
    plt.xlabel("Training Step")
    plt.ylabel("Average MMD of Generated Batch")
    plt.legend()
    plt.savefig(os.path.join(save_path, "3_dimension_locking.png"))
    plt.close()

def plot_4_action_minimization(logs, baseline_ensemble, save_path):
    """
    This plot requires running the 'bd' experiment.
    The 'avg_energy' column *is* the BD Action S_BD.
    """
    if 'avg_energy' not in logs.columns:
        print("Skipping Plot 4: 'avg_energy' not in logs.")
        return

    # Calculate baseline BD action
    print("Analyzing baseline ensemble for BD action...")
    baseline_stats = analyze_ensemble(baseline_ensemble, 'bd')
    avg_random_bd = np.mean(baseline_stats['bds'])

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=logs, x='step', y='avg_energy', label="GFN-Generated S_BD")
    plt.axhline(y=avg_random_bd, color='r', linestyle='--', label=f"Avg. Random S_BD ({avg_random_bd:.2f})")
    plt.title("Plot 4: Action Minimization (BD Experiment)")
    plt.xlabel("Training Step")
    plt.ylabel("Average Benincasa-Dowker Action (S_BD)")
    plt.legend()
    plt.savefig(os.path.join(save_path, "4_action_minimization.png"))
    plt.close()
    return avg_random_bd

def plot_5_money_shot(ensemble, baseline_ensemble, save_path):
    """
    The "Money Shot" plot.
    Requires ensemble from 'bd' experiment.
    """
    print("Analyzing GFN ensemble for MMD...")
    gfn_stats = analyze_ensemble(ensemble, 'mmd')
    gfn_mmds = gfn_stats['mmds']

    print("Analyzing baseline ensemble for MMD...")
    baseline_stats = analyze_ensemble(baseline_ensemble, 'mmd')
    baseline_mmds = baseline_stats['mmds']

    df1 = pd.DataFrame({'MMD': gfn_mmds, 'Distribution': 'GFN (Trained on BD Action)'})
    df2 = pd.DataFrame({'MMD': baseline_mmds, 'Distribution': 'Random (Baseline)'})
    plot_df = pd.concat([df1, df2])

    plt.figure(figsize=(12, 7))
    sns.histplot(data=plot_df, x='MMD', hue='Distribution', kde=True,
                 element="step", common_norm=False, stat="density")
    plt.title("Plot 5: Emergent Dimension (The 'Money Shot')")
    plt.xlabel("Myrheim-Meyer Dimension (MMD)")
    plt.ylabel("Density")
    plt.axvline(x=4.0, color='k', linestyle=':', label="d=4")
    plt.legend()
    plt.savefig(os.path.join(save_path, "5_money_shot_histogram.png"))
    plt.close()

def main_analysis():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_dir", type=str, required=True,
        help="Path to the experiment run directory (e.g., 'experiment_results/run_N15_beta1_bd')"
    )
    parser.add_argument(
        "--baseline_file", type=str, required=True,
        help="Path to the baseline ensemble (e.g., 'baseline_ensemble_n40.pkl')"
    )
    parser.add_argument(
        "--reward_type", type=str, required=True, choices=["bd", "mmd"],
        help="The reward type this experiment was run with."
    )
    args = parser.parse_args()

    plot_save_dir = os.path.join(args.run_dir, "plots")
    os.makedirs(plot_save_dir, exist_ok=True)

    print(f"Loading results from {args.run_dir}")
    logs, ensemble = load_results(args.run_dir)

    print(f"Loading baseline from {args.baseline_file}")
    baseline_ensemble = load_baseline(args.baseline_file)

    print("Generating plots...")

    # --- Sanity Checks ---
    plot_1_loss(logs, plot_save_dir)
    plot_2_reward(logs, plot_save_dir)
    print("Generated Plots 1 & 2.")

    # --- Main Paper Plots ---
    if args.reward_type == 'mmd':
        plot_3_dim_locking(logs, ensemble, plot_save_dir)
        print("Generated Plot 3.")

    elif args.reward_type == 'bd':
        plot_4_action_minimization(logs, baseline_ensemble, plot_save_dir)
        print("Generated Plot 4.")

        plot_5_money_shot(ensemble, baseline_ensemble, plot_save_dir)
        print("Generated Plot 5 (The 'Money Shot').")

    print(f"All plots saved to {plot_save_dir}")
    print("\nNote: Plots 6 & 7 (Scaling & Ablation) require running `train.py`")
    print("multiple times with different --N and --beta arguments,")
    print("then aggregating the results with a custom script.")

if __name__ == "__main__":
    main_analysis()
