#
# gflownet-causet-genesis/train.py
#
# --- THIS FILE IS CORRECTED ---
#
# Fixes applied:
# 1. Simplified CausetPolicyEstimator (no overrides).
# 2. Passed annealed 'current_temp' and 'current_epsilon'
#    directly into sampler.sample_trajectories() as
#    keyword arguments, which the base class handles.
#
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import pandas as pd
import numpy as np
import pickle
import argparse
import os
from tqdm import tqdm
import time
import math

from gfn.gflownet.trajectory_balance import TBGFlowNet
from gfn.estimators import DiscretePolicyEstimator
from gfn.samplers import Sampler

from causet_env import CausalSetEnv
from causet_reward import CausalSetRewardProxy


class CausetPolicyNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.input_dim = state_dim

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: State tensor of shape [batch, state_dim]
        Returns:
            Logits of shape [batch, action_dim]
        """
        return self.network(x)


class CausetPolicyEstimator(DiscretePolicyEstimator):
    def __init__(self, env: CausalSetEnv, hidden_dim: int = 256):

        self.env = env

        module = CausetPolicyNetwork(
            state_dim=env.state_dim,
            hidden_dim=hidden_dim,
            action_dim=env.n_actions
        )

        # Use the base class __init__
        super().__init__(
            module=module,
            n_actions=env.n_actions,
            preprocessor=None,
            is_backward=False
        )

        self.to(env.device)

    # No overrides are needed. The base class methods will be used.

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=10,
                       help="Max causet size (start smaller for debugging)")
    parser.add_argument(
        "--reward_type", type=str, default="bd", choices=["bd", "mmd"],
        help="Reward: 'bd' (Benincasa-Dowker) or 'mmd' (Myrheim-Meyer)"
    )
    parser.add_argument("--beta", type=float, default=1.0,
                       help="Inverse temperature (beta)")

    parser.add_argument("--num_steps", type=int, default=5_000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--device", type=str,
                       default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--epsilon_start", type=float, default=0.3)
    parser.add_argument("--epsilon_end", type=float, default=0.05)
    parser.add_argument("--epsilon_decay_steps", type=int, default=3000)
    parser.add_argument("--temp_start", type=float, default=2.0)
    parser.add_argument("--temp_end", type=float, default=1.0)
    parser.add_argument("--temp_decay_steps", type=int, default=3000)

    parser.add_argument("--output_dir", type=str, default="experiment_results")
    parser.add_argument("--run_name", type=str, default="run_N10_beta1_bd")

    return parser.parse_args()


def main():
    args = parse_args()

    run_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Saving results to: {run_dir}")
    print(f"Using device: {args.device}")

    print("Initializing environment and reward proxy...")
    # Use the corrected causet_env.py
    proxy = CausalSetRewardProxy(reward_type=args.reward_type)
    env = CausalSetEnv(max_nodes=args.N, proxy=proxy, device=args.device)

    print("Initializing policy network...")
    # Initialize the simplified estimator
    pf = CausetPolicyEstimator(
        env=env,
        hidden_dim=args.hidden_dim
    )

    pb = pf

    print("Initializing GFlowNet...")
    gflownet = TBGFlowNet(pf=pf, pb=pb)

    optimizer = optim.Adam(pf.parameters(), lr=args.lr)

    print(f"\nStarting training for {args.num_steps} steps...")
    print(f"Batch size: {args.batch_size}")
    print(f"Max nodes: {args.N}")
    print(f"Reward type: {args.reward_type}")

    log_data = []
    start_time = time.time()

    # Create sampler *once* outside the loop
    sampler = Sampler(estimator=pf)

    for step in range(args.num_steps):  # Direct range instead of tqdm
        if step % 100 == 0:
            print(f"\n--- Step {step} ---")

        try:
            # Anneal exploration parameters
            progress = min(step / args.epsilon_decay_steps, 1.0)
            current_epsilon = args.epsilon_start + \
                            (args.epsilon_end - args.epsilon_start) * progress

            progress_temp = min(step / args.temp_decay_steps, 1.0)
            current_temp = args.temp_start + \
                          (args.temp_end - args.temp_start) * progress_temp

            # Training step
            pf.train()
            optimizer.zero_grad()

            # Pass annealed values as keyword arguments
            # The base DiscretePolicyEstimator handles these.
            trajectories = sampler.sample_trajectories(
                env=env,
                n=args.batch_size,
                temperature=current_temp,
                epsilon=current_epsilon
            )

            # Calculate loss
            # With the corrected causet_env.py, this should work
            loss = gflownet.loss(env, trajectories)

            if not loss.isfinite():
                print(f"\nWarning: Non-finite loss at step {step}")
                continue

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pf.parameters(), 1.0)
            optimizer.step()

            # Logging
            if step % 100 == 0 or step == args.num_steps - 1:
                all_states_tensor = trajectories.states.tensor

                if all_states_tensor.dim() == 3:
                    batch_size = all_states_tensor.shape[1]
                    terminating_states_list = []

                    is_sink = trajectories.next_states.is_sink_state

                    for b in range(batch_size):
                        sink_idx = is_sink[:, b].nonzero(as_tuple=True)[0]
                        if len(sink_idx) > 0:
                            last_idx = sink_idx[0]
                            terminating_states_list.append(all_states_tensor[last_idx, b, :])

                    if not terminating_states_list:
                        print("Warning: No valid terminating states found in batch.")
                        continue

                    terminating_states_tensor = torch.stack(terminating_states_list)
                else:
                    terminating_states_tensor = all_states_tensor[trajectories.is_terminal]

                energies_list = []
                rewards_list = []
                valid_count = 0

                for i in range(terminating_states_tensor.shape[0]):
                    state = terminating_states_tensor[i]
                    n = int(state[0].item())

                    if n == args.N:
                        g = env._state_to_graph(state)
                        energy = proxy.get_energy(g)

                        if energy < 1e9:
                            energies_list.append(energy)
                            rewards_list.append(math.exp(-args.beta * energy))
                            valid_count += 1

                avg_energy = np.mean(energies_list) if energies_list else 1e9
                avg_reward = np.mean(rewards_list) if rewards_list else 0.0

                unique_graphs = set()
                for i in range(terminating_states_tensor.shape[0]):
                    state = terminating_states_tensor[i]
                    n = int(state[0].item())
                    if n == args.N:
                        edges = tuple(state[1:].cpu().numpy())
                        unique_graphs.add(edges)

                diversity = len(unique_graphs) / max(valid_count, 1)

                log_data.append({
                    'step': step,
                    'loss': loss.item(),
                    'avg_energy': avg_energy,
                    'avg_reward': avg_reward,
                    'diversity': diversity,
                    'epsilon': current_epsilon,
                    'temperature': current_temp,
                    'elapsed_time_sec': time.time() - start_time
                })

                print(f"Step {step}: loss={loss.item():.2e}, avg_e={avg_energy:.2e}, "
                      f"avg_r={avg_reward:.4f}, diversity={diversity:.3f}, valid={valid_count}/{terminating_states_tensor.shape[0]}")

        except Exception as e:
            print(f"\nError during step {step}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\nTraining finished.")
    df = pd.DataFrame(log_data)
    df.to_csv(os.path.join(run_dir, "training_log.csv"), index=False)
    print(f"Log saved to {os.path.join(run_dir, 'training_log.csv')}")

    torch.save(pf.state_dict(), os.path.join(run_dir, "policy_model.pt"))
    print(f"Model saved to {os.path.join(run_dir, 'policy_model.pt')}")


if __name__ == '__main__':
    main()
