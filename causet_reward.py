#
# gflownet-causet-genesis/causet_reward.py
# OPTIMIZED VERSION - Batched GPU operations
#
import torch
import networkx as nx
import numpy as np
import random
from scipy.optimize import root_scalar
from scipy.special import gamma


class CausalSetRewardProxy:
    """
    Reward Proxy with batched GPU operations for BD action.
    MMD kept for ablation studies (CPU-only).
    """
    def __init__(self, reward_type: str, target_dim: float = 4.0, device='cuda'):
        if reward_type not in ['mmd', 'bd']:
            raise ValueError("reward_type must be 'mmd' or 'bd'")
        self.reward_type = reward_type
        self.target_dim = target_dim
        self.device = device

        if reward_type == 'mmd':
            self.target_ratio = self._mmd_ratio_func_single(self.target_dim)
            print(f"Initialized RewardProxy with type: {self.reward_type} (CPU-only)")
            print(f"Target dimension {self.target_dim} corresponds to C2/C1^2 ratio: {self.target_ratio:.4f}")
        else:
            print(f"Initialized RewardProxy with type: {self.reward_type} (GPU-accelerated)")

    def get_energy(self, g: nx.DiGraph) -> float:
        """Legacy interface for single graph (for analysis)"""
        if g.number_of_nodes() < 2:
            return 1e10

        if self.reward_type == 'mmd':
            return self.get_mmd_energy(g)
        elif self.reward_type == 'bd':
            return self.get_bd_energy(g)

    # --- BATCHED BD ACTION (GPU-ACCELERATED) ---

    def get_bd_energy_batched(self, state_tensors: torch.Tensor, max_nodes: int) -> torch.Tensor:
        """
        Compute BD action for a batch of states using pure PyTorch.

        Args:
            state_tensors: [batch_size, state_dim] - state vectors
            max_nodes: maximum number of nodes

        Returns:
            energies: [batch_size] - BD action values
        """
        batch_size = state_tensors.shape[0]
        device = state_tensors.device

        # Extract n values
        n_values = state_tensors[:, 0].long()  # [batch_size]

        # Initialize energies with high penalty for incomplete causets
        energies = torch.full((batch_size,), 1e10, device=device, dtype=torch.float32)

        # Only compute for complete causets (n == max_nodes)
        valid_mask = (n_values == max_nodes)

        if not valid_mask.any():
            return energies

        # Process only valid states
        valid_states = state_tensors[valid_mask]  # [n_valid, state_dim]
        n_valid = valid_mask.sum().item()

        # Extract adjacency matrices from state vectors
        # State format: [n, edge_0_1, edge_0_2, ..., edge_{n-2}_{n-1}]
        max_edges = (max_nodes * (max_nodes - 1)) // 2

        # Build adjacency matrices efficiently
        adj_matrices = self._build_adjacency_matrices_batched(
            valid_states[:, 1:1+max_edges], max_nodes
        )  # [n_valid, max_nodes, max_nodes]

        # Compute transitive closure (causal matrices)
        causal_matrices = self._transitive_closure_batched(adj_matrices)

        # Make reflexive
        eye = torch.eye(max_nodes, device=device, dtype=torch.bool)
        causal_matrices = causal_matrices | eye.unsqueeze(0)

        # Compute BD action for each valid state
        bd_actions = self._compute_bd_action_batched(
            causal_matrices.float(), max_nodes
        )  # [n_valid]

        # Fill in results
        energies[valid_mask] = bd_actions

        return energies

    def _build_adjacency_matrices_batched(self, edge_vectors: torch.Tensor,
                                         max_nodes: int) -> torch.Tensor:
        """
        Convert edge vectors to adjacency matrices.

        Args:
            edge_vectors: [batch_size, max_edges] - binary edge indicators
            max_nodes: number of nodes

        Returns:
            adj_matrices: [batch_size, max_nodes, max_nodes] - adjacency matrices
        """
        batch_size = edge_vectors.shape[0]
        device = edge_vectors.device

        # Initialize adjacency matrices
        adj = torch.zeros(batch_size, max_nodes, max_nodes,
                         device=device, dtype=torch.bool)

        # Map flat edge index to (i, j) pairs
        edge_idx = 0
        for i in range(max_nodes):
            for j in range(i + 1, max_nodes):
                if edge_idx < edge_vectors.shape[1]:
                    adj[:, i, j] = edge_vectors[:, edge_idx] > 0.5
                edge_idx += 1

        return adj

    def _transitive_closure_batched(self, adj_matrices: torch.Tensor) -> torch.Tensor:
        """
        Compute transitive closure using Warshall's algorithm (batched).

        Args:
            adj_matrices: [batch_size, n, n] - adjacency matrices

        Returns:
            closures: [batch_size, n, n] - transitive closures
        """
        batch_size, n, _ = adj_matrices.shape

        # Start with adjacency matrix
        tc = adj_matrices.clone()

        # Warshall's algorithm
        for k in range(n):
            # tc[i,j] |= tc[i,k] & tc[k,j]
            tc = tc | (tc[:, :, k:k+1] & tc[:, k:k+1, :])

        return tc

    def _compute_bd_action_batched(self, causal_matrices: torch.Tensor,
                                   n: int) -> torch.Tensor:
        """
        Compute BD action from causal matrices (batched).

        Args:
            causal_matrices: [batch_size, n, n] - reflexive causal matrices
            n: number of nodes

        Returns:
            actions: [batch_size] - BD action values
        """
        batch_size = causal_matrices.shape[0]

        # Compute interval sizes: C @ C
        # interval_sizes[b, i, j] = number of elements in I[i,j]
        interval_sizes = torch.matmul(causal_matrices, causal_matrices)  # [batch_size, n, n]

        # Flatten to count occurrences
        interval_sizes_flat = interval_sizes.reshape(batch_size, -1)  # [batch_size, n*n]

        # Count N_k for each batch element
        # We need counts for sizes 2, 3, 4, 5 (corresponding to k=0,1,2,3)
        N_0 = (interval_sizes_flat == 2).sum(dim=1).float()  # [batch_size]
        N_1 = (interval_sizes_flat == 3).sum(dim=1).float()
        N_2 = (interval_sizes_flat == 4).sum(dim=1).float()
        N_3 = (interval_sizes_flat == 5).sum(dim=1).float()

        # S_BD = n - N_0 + 9*N_1 - 16*N_2 + 8*N_3
        actions = n - N_0 + 9*N_1 - 16*N_2 + 8*N_3

        return actions

    # --- LEGACY SINGLE-GRAPH BD ACTION ---

    def get_bd_energy(self, g: nx.DiGraph) -> float:
        """Legacy BD action for single graph (used in analysis)"""
        n = g.number_of_nodes()
        if n < 4:
            return 1e10

        tc_g = nx.transitive_closure(g, reflexive=True)
        node_list = sorted(g.nodes())
        C_matrix = nx.adjacency_matrix(tc_g, nodelist=node_list).toarray().astype(np.int32)

        interval_sizes = C_matrix @ C_matrix
        counts = np.bincount(interval_sizes.ravel())

        N_0 = counts[2] if len(counts) > 2 else 0
        N_1 = counts[3] if len(counts) > 3 else 0
        N_2 = counts[4] if len(counts) > 4 else 0
        N_3 = counts[5] if len(counts) > 5 else 0

        action_value = float(n - N_0 + (9 * N_1) - (16 * N_2) + (8 * N_3))
        return action_value

    # --- MMD (CPU-ONLY, FOR ABLATION) ---

    def _mmd_ratio_func_single(self, d):
        if d <= 1: return 0.0
        return (gamma(d + 1) * gamma(d / 2)) / (4 * gamma(d) * gamma(1.5 * d))

    def solve_mmd_from_ratio(self, ratio):
        if ratio <= 0 or not np.isfinite(ratio):
            return 1.01

        try:
            f = lambda d: self._mmd_ratio_func_single(d) - ratio
            sol = root_scalar(f, bracket=[1.01, 10.0], method='bisect')
            return sol.root
        except ValueError:
            if ratio > self._mmd_ratio_func_single(10.0):
                return 10.0
            else:
                return 1.01
        except Exception:
            return 1.01

    def calculate_avg_mmd_ratio(self, g: nx.DiGraph, num_samples=200):
        nodes = list(g.nodes)
        n = len(nodes)
        if n < 2:
            return 0.0

        tc_inclusive = nx.transitive_closure(g, reflexive=True)
        tc_exclusive = nx.transitive_closure(g, reflexive=False)
        ratios = []

        for _ in range(num_samples):
            x, y = random.sample(nodes, 2)

            if tc_inclusive.has_edge(y, x): x, y = y, x
            elif not tc_inclusive.has_edge(x, y): continue

            interval_nodes = {
                z for z in nodes
                if tc_exclusive.has_edge(x, z) and tc_exclusive.has_edge(z, y)
            }
            C_1 = len(interval_nodes)

            if C_1 < 2: continue

            C_2 = 0
            interval_list = list(interval_nodes)
            for i in range(C_1):
                for j in range(i + 1, C_1):
                    u, v = interval_list[i], interval_list[j]
                    if tc_inclusive.has_edge(u, v) or tc_inclusive.has_edge(v, u):
                        C_2 += 1

            ratios.append(C_2 / (C_1**2))

        return np.mean(ratios) if ratios else 0.0

    def get_mmd_energy(self, g: nx.DiGraph) -> float:
        try:
            avg_ratio = self.calculate_avg_mmd_ratio(g)
            avg_mmd = self.solve_mmd_from_ratio(avg_ratio)
            energy = (avg_mmd - self.target_dim)**2
            return energy
        except Exception:
            return 1e10

    def calculate_avg_mmd(self, g: nx.DiGraph, num_samples=200):
        avg_ratio = self.calculate_avg_mmd_ratio(g, num_samples)
        return self.solve_mmd_from_ratio(avg_ratio)
