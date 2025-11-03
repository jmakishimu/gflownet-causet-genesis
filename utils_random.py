import networkx as nx
import random
import pickle
from tqdm import tqdm

def generate_random_causet(n_nodes: int, p: float = 0.5) -> nx.DiGraph:
    """
    Generates a random causet using a simple percolation model.
    This serves as the "junk" baseline.
    """
    g = nx.DiGraph()
    g.add_nodes_from(range(n_nodes))

    # Generate random edges
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if random.random() < p:
                g.add_edge(i, j)

    # Enforce transitivity
    tc = nx.transitive_closure(g)
    return tc

def generate_baseline_ensemble(N: int, num_samples: int, p: float, filepath: str):
    """
    Generates and saves an ensemble of random causets.
    """
    print(f"Generating {num_samples} random causets (N={N}, p={p})...")
    ensemble = [
        generate_random_causet(N, p) for _ in tqdm(range(num_samples))
    ]

    with open(filepath, 'wb') as f:
        pickle.dump(ensemble, f)
    print(f"Saved baseline ensemble to {filepath}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=40)
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--p", type=float, default=0.3)
    parser.add_argument("--out_file", type=str, default="baseline_ensemble_n40.pkl")
    args = parser.parse_args()

    generate_baseline_ensemble(args.N, args.num_samples, args.p, args.out_file)
