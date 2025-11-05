#!/usr/bin/env python
"""
Test if trajectories are being sampled correctly
"""
import sys
import torch
import torch.nn as nn
from gfn.samplers import Sampler
from gfn.estimators import DiscretePolicyEstimator

from causet_env import CausalSetEnv
from causet_reward import CausalSetRewardProxy

class SimplePolicy(nn.Module):
    """Simple MLP policy"""
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.input_dim = input_dim  # Required by torchgfn
        self.output_dim = output_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)

def test_sampling():
    print("="*60, flush=True)
    print("TESTING TRAJECTORY SAMPLING", flush=True)
    print("="*60, flush=True)

    # Setup
    N = 40
    batch_size = 4  # Small batch for testing

    print(f"\nSetting up environment (N={N})...", flush=True)
    proxy = CausalSetRewardProxy(reward_type='bd')
    env = CausalSetEnv(max_nodes=N, proxy=proxy, device='cpu')

    print(f"Environment info:", flush=True)
    print(f"  max_nodes: {env.max_nodes}", flush=True)
    print(f"  n_actions: {env.n_actions}", flush=True)
    print(f"  state_dim: {env.state_dim}", flush=True)

    # Create a simple policy
    print(f"\nCreating policy network...", flush=True)
    module = SimplePolicy(
        input_dim=env.state_dim,
        output_dim=env.n_actions
    )

    pf = DiscretePolicyEstimator(
        module=module,
        n_actions=env.n_actions,
        is_backward=False,
        preprocessor=None
    )

    # Create sampler with explicit parameters
    print(f"Creating sampler...", flush=True)
    # Check if we can pass max_length to the Sampler constructor
    try:
        sampler = Sampler(estimator=pf)
        print(f"Sampler created (will use default max_length)", flush=True)
    except Exception as e:
        print(f"Failed to create sampler: {e}", flush=True)
        sys.exit(1)

    # Sample trajectories
    desired_max_length = N * (N + 1) + 10  # Should be ~40 for N=5
    print(f"\nSampling {batch_size} trajectories...", flush=True)
    print(f"Desired max_length: {desired_max_length}", flush=True)
    print(f"Trying to pass max_length parameter...", flush=True)

    # Try different parameter names
    for param_name in ['max_length', 'max_len', 'n_steps']:
        try:
            print(f"  Trying parameter '{param_name}'...", flush=True)
            trajectories = sampler.sample_trajectories(
                env=env,
                n=batch_size,
                **{param_name: desired_max_length}
            )
            print(f"  ✓ '{param_name}' worked!", flush=True)
            break
        except TypeError as e:
            if 'unexpected keyword argument' in str(e):
                print(f"  ✗ '{param_name}' not supported", flush=True)
                continue
            else:
                raise
    else:
        # None worked, try without parameter
        print(f"  No max_length parameter supported, using default...", flush=True)
        trajectories = sampler.sample_trajectories(
            env=env,
            n=batch_size
        )

    print("✓ Sampling succeeded!", flush=True)

    # Analyze trajectories
    print(f"\n" + "="*60, flush=True)
    print("TRAJECTORY ANALYSIS", flush=True)
    print("="*60, flush=True)

    all_states = trajectories.states.tensor
    print(f"\nTrajectories tensor shape: {all_states.shape}", flush=True)

    if all_states.dim() == 3:
        n_steps, batch_size_actual, state_dim = all_states.shape
        print(f"  n_steps: {n_steps}", flush=True)
        print(f"  batch_size: {batch_size_actual}", flush=True)
        print(f"  state_dim: {state_dim}", flush=True)

        # Analyze each trajectory
        print(f"\nPer-trajectory analysis:", flush=True)
        completed_count = 0

        for b in range(batch_size_actual):
            traj_states = all_states[:, b, :]
            non_sink = traj_states[:, 0] != -1

            if non_sink.any():
                traj_length = non_sink.sum().item()
                final_n = int(traj_states[non_sink][-1, 0].item())
                completed = "✓ COMPLETE" if final_n == N else f"✗ incomplete (n={final_n})"

                print(f"  Trajectory {b}: length={traj_length:2d}, final_n={final_n}, {completed}", flush=True)

                if final_n == N:
                    completed_count += 1
            else:
                print(f"  Trajectory {b}: EMPTY (all sink states)", flush=True)

        print(f"\nSummary: {completed_count}/{batch_size_actual} trajectories completed", flush=True)

        # Show first trajectory in detail
        print(f"\n" + "-"*60, flush=True)
        print("First trajectory detail:", flush=True)
        print("-"*60, flush=True)
        traj_0 = all_states[:, 0, :]
        for t in range(min(20, n_steps)):
            n_val = traj_0[t, 0].item()
            if n_val != -1:
                # Show edges too
                edges = traj_0[t, 1:].nonzero().squeeze()
                print(f"  Step {t:2d}: n={int(n_val)}, edges={edges.tolist() if edges.numel() > 0 else []}", flush=True)
            else:
                print(f"  Step {t:2d}: SINK (trajectory ended)", flush=True)
                break

        # Check if the sampler is stopping early
        print(f"\n" + "-"*60, flush=True)
        print("Diagnosis:", flush=True)
        print("-"*60, flush=True)
        print(f"The sampler only took {n_steps} steps total.", flush=True)
        print(f"To reach N={N}, we need at least {N} steps (one per node).", flush=True)
        print(f"\nPossible issues:", flush=True)
        print(f"  1. Sampler has a hardcoded max_length limit", flush=True)
        print(f"  2. Environment is incorrectly marking states as terminal", flush=True)
        print(f"  3. Forward masks are incorrect (no valid actions)", flush=True)

    print(f"\n" + "="*60, flush=True)
    if completed_count > 0:
        print(f"SUCCESS: {completed_count} trajectories completed!", flush=True)
    else:
        print(f"PROBLEM: NO trajectories reached terminal state!", flush=True)
        print(f"This explains why training isn't working.", flush=True)
    print("="*60, flush=True)

if __name__ == "__main__":
    test_sampling()
