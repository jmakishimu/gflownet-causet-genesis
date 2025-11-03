#
# gflownet-causet-genesis/causet_reward.py
#
import math
import networkx as nx
import numpy as np
import random
from scipy.optimize import root_scalar # (Fix #1 Applied) Import root_scalar
from scipy.special import gamma

class CausalSetRewardProxy:
    """
    Implements the Reward Proxy (Phase 3).

    - Fix #3: Correct Benincasa-Dowker matrix product (C @ C)
    - Fix #5: (MODIFIED) Robust MMD solver using scipy.optimize.root_scalar
    - MMD Flaw Fix: Correctly handles sparse/1D causets by falling
                    back to an average ratio of 0.0 (not target_ratio).
    """
    def __init__(self, reward_type: str, target_dim: float = 4.0):
        if reward_type not in ['mmd', 'bd']:
            raise ValueError("reward_type must be 'mmd' or 'bd'")
        self.reward_type = reward_type
        self.target_dim = target_dim

        # Pre-calculate the target ratio for d=4
        self.target_ratio = self._mmd_ratio_func_single(self.target_dim)

        # (Fix #1 Applied) Remove pre-calculated mmd_func and ranges,
        # as we are now using a true solver.

        print(f"Initialized RewardProxy with type: {self.reward_type}")
        print(f"Target dimension {self.target_dim} corresponds to C2/C1^2 ratio: {self.target_ratio:.4f}")

    def get_energy(self, g: nx.DiGraph) -> float:
        """
        Returns the energy (e.g., Action S_BD or (MMD-4)^2) for the GFlowNet.
        Low energy = high reward.
        """
        if g.number_of_nodes() < 2: # Trivial graph
            return 1e10

        if self.reward_type == 'mmd':
            return self.get_mmd_energy(g)
        elif self.reward_type == 'bd':
            return self.get_bd_energy(g)

    # --- Reward A: Myrheim-Meyer Dimension (MMD) ---

    def _mmd_ratio_func_single(self, d):
        """ (Fix #5) The theoretical C2/C1^2 ratio for a given d. """
        if d <= 1: return 0.0 # Ratio is 0 for d=1
        return (gamma(d + 1) * gamma(d / 2)) / (4 * gamma(d) * gamma(1.5 * d))

    def solve_mmd_from_ratio(self, ratio):
        """
        (Fix #1 Applied) Robustly solves for d given a C2/C1^2 ratio
        using scipy.optimize.root_scalar.
        """
        if ratio <= 0 or not np.isfinite(ratio):
            return 1.01 # Return a value near 1D

        try:
            # Define the function to solve: f(d) = 0
            f = lambda d: self._mmd_ratio_func_single(d) - ratio

            # Solve for d in the bracket [1.01, 10.0]
            # bisect is robust and guaranteed to stay in bounds.
            sol = root_scalar(f, bracket=[1.01, 10.0], method='bisect')
            return sol.root
        except ValueError:
            # If ratio is outside the range produced by [1.01, 10.0],
            # root_scalar will raise a ValueError.
            # Return the closest boundary.
            if ratio > self.target_ratio: # d > 4
                return 10.0
            else: # d < 4
                return 1.01
        except Exception:
            # Catch other potential errors
            return 1.01 # Fallback

    def calculate_avg_mmd_ratio(self, g: nx.DiGraph, num_samples=200):
        """
        (MMD Flaw Fix) Calculates the average C2/C1^2 ratio.
        Includes C2=0 cases and falls back to 0.0 if no valid
        C1>=2 intervals are found.
        """
        nodes = list(g.nodes)
        n = len(nodes)
        if n < 2:
            return 0.0

        tc = nx.transitive_closure(g, reflexive=True)
        ratios = []

        for _ in range(num_samples):
            x, y = random.sample(nodes, 2)

            if tc.has_edge(y, x): x, y = y, x # Ensure x < y
            elif not tc.has_edge(x, y): continue # Spacelike

            # Vectorized interval check
            interval_nodes = {z for z in nodes if tc.has_edge(x, z) and tc.has_edge(z, y)}
            C_1 = len(interval_nodes)

            if C_1 < 2: continue # Not a valid interval, skip sample (C_1 is inclusive, so C_1 >= 2)

            C_2 = 0
            interval_list = list(interval_nodes)
            for i in range(C_1):
                for j in range(i + 1, C_1):
                    u, v = interval_list[i], interval_list[j]
                    if tc.has_edge(u, v) or tc.has_edge(v, u):
                        C_2 += 1

            # Add the ratio, *including* C_2 = 0 cases
            ratios.append(C_2 / (C_1**2))

        # (THE FIX) If 'ratios' is empty (e.g., a 1D chain where
        # C1>=2 but C2=0 always, or just no C1>=2 intervals found),
        # return 0.0, not the target_ratio.
        return np.mean(ratios) if ratios else 0.0

    def get_mmd_energy(self, g: nx.DiGraph) -> float:
        """
        Energy = (avg_MMD(c) - 4.0)^2

        This now correctly calculates energy based on the average ratio,
        preventing the 1D-chain-gets-zero-energy flaw.
        """
        try:
            # 1. Get the average ratio. This will be 0.0 for 1D chains.
            avg_ratio = self.calculate_avg_mmd_ratio(g)

            # 2. Solve for the dimension *once* based on the average ratio.
            avg_mmd = self.solve_mmd_from_ratio(avg_ratio)

            # 3. Calculate energy.
            # If avg_mmd = 1.01 (from ratio 0.0), energy will be high:
            # (1.01 - 4.0)^2 >> 0
            energy = (avg_mmd - self.target_dim)**2
            return energy
        except Exception as e:
            return 1e10 # High energy if calc fails

    def calculate_avg_mmd(self, g: nx.DiGraph, num_samples=200):
        """
        Helper function for analysis.py.
        Returns the calculated average MMD (not the energy).
        """
        avg_ratio = self.calculate_avg_mmd_ratio(g, num_samples)
        return self.solve_mmd_from_ratio(avg_ratio)

    # --- Reward B: Benincasa-Dowker (BD) Action ---

    def get_bd_energy(self, g: nx.DiGraph) -> float:
        """
        Reward B: The "Parsimonious" SOTA (Benincasa-Dowker Action)
        Energy = S_BD[c]

        (Fix #3) This implementation is vectorized with the *correct*
        matrix multiplication (C @ C) to find causal intervals.
        """
        n = g.number_of_nodes()
        if n < 4: # BD action is trivial for < 4 nodes
            return 1e10

        # 1. Get Causal Matrix C (reflexive)
        C_matrix = nx.transitive_closure_matrix(g, reflexive=True)
        C_matrix = C_matrix.toarray().astype(np.int32)

        # 2. Compute all N^2 inclusive interval sizes |I[i, j]|
        # (Fix #3) The (i, j) entry of (C @ C) is sum_k C_ik * C_kj.
        interval_sizes = C_matrix @ C_matrix

        # 3. INTRINSIC PART: Count the N_k
        counts = np.bincount(interval_sizes.ravel())

        # 4. EXTRINSIC PART: Apply 4D coefficients
        N_0 = counts[2] if len(counts) > 2 else 0 # k=0 -> size k+2 = 2
        N_1 = counts[3] if len(counts) > 3 else 0 # k=1 -> size k+2 = 3
        N_2 = counts[4] if len(counts) > 4 else 0 # k=2 -> size k+2 = 4
        N_3 = counts[5] if len(counts) > 5 else 0 # k=3 -> size k+2 = 5

        # S_4D_action = n - N_0 + 9*N_1 - 16*N_2 + 8*N_3
        action_value = float(n - N_0 + (9 * N_1) - (16 * N_2) + (8 * N_3))

        return action_value
