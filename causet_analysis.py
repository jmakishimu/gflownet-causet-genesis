#!/usr/bin/env python3
"""
Explore the best sampled causets and analyze their properties
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from causet_reward import CausalSetRewardProxy
from tqdm import tqdm
import argparse

def analyze_ensemble(ensemble_path: str, top_k: int = 10):
    """
    Load ensemble and analyze the best causets
    """
    print(f"Loading ensemble from {ensemble_path}...")
    with open(ensemble_path, 'rb') as f:
        ensemble = pickle.load(f)

    print(f"Ensemble size: {len(ensemble)} graphs")

    # Initialize reward proxies
    bd_proxy = CausalSetRewardProxy(reward_type='bd', device='cpu')
    mmd_proxy = CausalSetRewardProxy(reward_type='mmd', device='cpu')

    # Calculate both BD and MMD for all graphs
    print("\nCalculating BD action and MMD for all graphs...")
    results = []

    for i, g in enumerate(tqdm(ensemble)):
        try:
            bd = bd_proxy.get_bd_energy(g)
            mmd = mmd_proxy.calculate_avg_mmd(g, num_samples=200)
            n = g.number_of_nodes()
            m = g.number_of_edges()

            results.append({
                'index': i,
                'graph': g,
                'bd': bd,
                'mmd': mmd,
                'n': n,
                'm': m
            })
        except Exception as e:
            print(f"Error processing graph {i}: {e}")
            continue

    # Sort by BD action (lower is better)
    results.sort(key=lambda x: abs(x['bd']))

    # Print statistics
    print("\n" + "="*70)
    print("ENSEMBLE STATISTICS")
    print("="*70)

    bd_values = [r['bd'] for r in results]
    mmd_values = [r['mmd'] for r in results]

    print(f"\nBD Action Statistics:")
    print(f"  Mean: {np.mean(bd_values):.2f} ± {np.std(bd_values):.2f}")
    print(f"  Median: {np.median(bd_values):.2f}")
    print(f"  Min: {np.min(bd_values):.2f}")
    print(f"  Max: {np.max(bd_values):.2f}")

    print(f"\nMMD Statistics:")
    print(f"  Mean: {np.mean(mmd_values):.2f} ± {np.std(mmd_values):.2f}")
    print(f"  Median: {np.median(mmd_values):.2f}")
    print(f"  Target: 4.0 (for 4D spacetime)")

    # Analyze top-k best causets
    print("\n" + "="*70)
    print(f"TOP {top_k} CAUSETS (by |S_BD|)")
    print("="*70)
    print(f"\n{'Rank':<6} {'S_BD':<10} {'MMD':<10} {'Nodes':<8} {'Edges':<8} {'Edge Density':<12}")
    print("-"*70)

    for rank, r in enumerate(results[:top_k], 1):
        density = r['m'] / (r['n'] * (r['n'] - 1) / 2) if r['n'] > 1 else 0
        print(f"{rank:<6} {r['bd']:<10.2f} {r['mmd']:<10.2f} {r['n']:<8} {r['m']:<8} {density:<12.3f}")

    # Correlation analysis
    print("\n" + "="*70)
    print("CORRELATION ANALYSIS")
    print("="*70)

    correlation = np.corrcoef(bd_values, mmd_values)[0, 1]
    print(f"\nCorrelation between S_BD and MMD: {correlation:.3f}")

    if abs(correlation) < 0.3:
        print("→ WEAK correlation: BD and MMD capture different physics!")
    elif abs(correlation) < 0.7:
        print("→ MODERATE correlation: Some relationship exists")
    else:
        print("→ STRONG correlation: BD and MMD are closely related")

    # Visualize best causets
    visualize_best_causets(results[:top_k])

    # Create scatter plot
    create_scatter_plot(bd_values, mmd_values)

    return results


def visualize_best_causets(top_results, max_display: int = 5):
    """
    Visualize the structure of the best causets
    """
    n_display = min(len(top_results), max_display)

    fig, axes = plt.subplots(1, n_display, figsize=(4*n_display, 4))
    if n_display == 1:
        axes = [axes]

    for idx, (ax, r) in enumerate(zip(axes, top_results[:n_display])):
        g = r['graph']

        # Use hierarchical layout for causal structure
        pos = nx.spring_layout(g, seed=42)

        nx.draw_networkx_nodes(g, pos, node_color='lightblue',
                              node_size=500, ax=ax)
        nx.draw_networkx_edges(g, pos, edge_color='gray',
                              arrows=True, arrowsize=15, ax=ax)
        nx.draw_networkx_labels(g, pos, font_size=10, ax=ax)

        ax.set_title(f"Rank {idx+1}\nS_BD={r['bd']:.2f}, MMD={r['mmd']:.2f}",
                    fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('best_causets_visualization.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to best_causets_visualization.png")
    plt.close()


def create_scatter_plot(bd_values, mmd_values):
    """
    Create scatter plot of BD vs MMD
    """
    plt.figure(figsize=(10, 8))

    # Main scatter plot
    plt.scatter(bd_values, mmd_values, alpha=0.5, s=50, c=np.abs(bd_values),
               cmap='viridis')

    # Add target lines
    plt.axhline(y=4.0, color='red', linestyle='--', linewidth=2,
               label='Target MMD = 4 (4D spacetime)', alpha=0.7)
    plt.axvline(x=0, color='blue', linestyle='--', linewidth=2,
               label='Target S_BD = 0', alpha=0.7)

    # Mark best region
    plt.axvspan(-5, 5, alpha=0.1, color='green', label='Good S_BD region')
    plt.axhspan(3.0, 5.0, alpha=0.1, color='orange', label='Good MMD region')

    plt.xlabel('Benincasa-Dowker Action (S_BD)', fontsize=12)
    plt.ylabel('Myrheim-Meyer Dimension (MMD)', fontsize=12)
    plt.title('Emergent Dimension Analysis\nOptimized for S_BD, measuring MMD',
             fontsize=14, fontweight='bold')
    plt.colorbar(label='|S_BD|')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('bd_vs_mmd_scatter.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved scatter plot to bd_vs_mmd_scatter.png")
    plt.close()


def analyze_structure_properties(results, top_k=20):
    """
    Analyze structural properties of best causets
    """
    print("\n" + "="*70)
    print("STRUCTURAL PROPERTIES OF BEST CAUSETS")
    print("="*70)

    top_graphs = [r['graph'] for r in results[:top_k]]

    # Calculate various graph properties
    properties = {
        'max_chain_length': [],
        'max_antichain_size': [],
        'avg_degree': [],
        'transitivity': []
    }

    for g in top_graphs:
        # Maximum chain length (longest path)
        if g.number_of_nodes() > 0:
            try:
                max_chain = max(len(nx.dag_longest_path(g)) for _ in [0]) if g.number_of_edges() > 0 else g.number_of_nodes()
            except:
                max_chain = g.number_of_nodes()
            properties['max_chain_length'].append(max_chain)

            # Average degree
            degrees = [d for n, d in g.degree()]
            properties['avg_degree'].append(np.mean(degrees) if degrees else 0)

            # Transitivity (should be 1.0 for causets)
            properties['transitivity'].append(nx.transitivity(g))

    print(f"\nAverage properties of top {top_k} causets:")
    print(f"  Max chain length: {np.mean(properties['max_chain_length']):.2f} ± {np.std(properties['max_chain_length']):.2f}")
    print(f"  Avg degree: {np.mean(properties['avg_degree']):.2f} ± {np.std(properties['avg_degree']):.2f}")
    print(f"  Transitivity: {np.mean(properties['transitivity']):.4f}")

    if np.mean(properties['transitivity']) < 0.99:
        print("  ⚠️  WARNING: Some graphs may not be properly transitive!")


def main():
    parser = argparse.ArgumentParser(description="Explore best causets from trained ensemble")
    parser.add_argument("--ensemble_path", type=str,
                       default="experiment_results/improved_n10_bd_b24_h256_nocurr/final_ensemble.pkl",
                       help="Path to ensemble pickle file")
    parser.add_argument("--top_k", type=int, default=10,
                       help="Number of top causets to analyze")

    args = parser.parse_args()

    results = analyze_ensemble(args.ensemble_path, args.top_k)

    if results:
        analyze_structure_properties(results, min(20, len(results)))

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nKey questions answered:")
    print("1. What do best causets (S_BD ≈ 0) look like? → See visualization")
    print("2. What is their MMD? → See statistics and scatter plot")
    print("3. Is there correlation between S_BD and MMD? → See correlation analysis")
    print("\nGenerated files:")
    print("  - best_causets_visualization.png")
    print("  - bd_vs_mmd_scatter.png")


if __name__ == "__main__":
    main()
