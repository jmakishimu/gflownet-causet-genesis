#
# gflownet-causet-genesis/train.py
#
import torch
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
from custom_gfn import FactorPreAggAgent
from custom_gfn import FactorPreAggPolicy
from torch_geometric.data import Data, Batch
from torch_geometric.utils import from_networkx

from causet_env import CausalSetEnv
from causet_policy import GNNPolicy
from causet_reward import CausalSetRewardProxy

_collate_fn_call_count = 0
_last_collate_debug_print = 0

def collate_fn(states: list[tuple]) -> Batch:
    """Collates environment states into a PyG Batch with rich node features."""
    global _collate_fn_call_count, _last_collate_debug_print
    _collate_fn_call_count += 1
    current_time = time.time()
    if (current_time - _last_collate_debug_print > 5.0) or (_collate_fn_call_count <= 5):
        print(f"\n[DEBUG] collate_fn: Call #{_collate_fn_call_count}. Batch size: {len(states)}")
        if states:
            print(f"[DEBUG] collate_fn: First state: {states[0]}")
            if len(states) > 1:
                print(f"[DEBUG] collate_fn: Last state: {states[-1]}")
        _last_collate_debug_print = current_time

    data_list = []
    for (n, edges, _) in states:
        if n == 0:
            data = Data(
                x=torch.empty(0, 5).float(),
                edge_index=torch.empty(2, 0).long(),
                num_nodes=0
            )
        else:
            g = nx.DiGraph()
            g.add_nodes_from(range(n))
            g.add_edges_from(edges)

            try:
                tc = nx.transitive_closure_dag(g)
            except Exception:
                tc = nx.transitive_closure(g, reflexive=False)

            features = []
            for i in range(n):
                feat = [
                    float(i),
                    float(g.in_degree(i)),
                    float(g.out_degree(i)),
                    float(tc.in_degree(i)),
                    float(tc.out_degree(i))
                ]
                features.append(feat)

            data = from_networkx(g)
            data.x = torch.tensor(features, dtype=torch.float32)
            data.num_nodes = n

        data_list.append(data)

    return Batch.from_data_list(data_list)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=15, help="Max causet size (N)")
    parser.add_argument(
        "--reward_type", type=str, default="bd", choices=["bd", "mmd"],
        help="Reward: 'bd' (Benincasa-Dowker) or 'mmd' (Myrheim-Meyer)"
    )
    parser.add_argument("--beta", type=float, default=1.0, help="Inverse temperature (beta)")

    # Training hps
    parser.add_argument("--num_steps", type=int, default=10_000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Exploration parameters
    parser.add_argument("--epsilon_start", type=float, default=0.3, help="Initial epsilon for exploration")
    parser.add_argument("--epsilon_end", type=float, default=0.05, help="Final epsilon")
    parser.add_argument("--epsilon_decay_steps", type=int, default=5000, help="Steps to decay epsilon")
    parser.add_argument("--temp_start", type=float, default=2.0, help="Initial temperature")
    parser.add_argument("--temp_end", type=float, default=1.0, help="Final temperature")
    parser.add_argument("--temp_decay_steps", type=int, default=5000, help="Steps to decay temperature")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Entropy regularization coefficient")

    # Output
    parser.add_argument("--output_dir", type=str, default="experiment_results")
    parser.add_argument("--run_name", type=str, default="run_N15_beta1_bd")

    return parser.parse_args()


def main():
    args = parse_args()

    run_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Saving results to: {run_dir}")
    print(f"Using device: {args.device}")
    print(f"Exploration: epsilon {args.epsilon_start}->{args.epsilon_end}, temp {args.temp_start}->{args.temp_end}")

    # Setup components
    proxy = CausalSetRewardProxy(reward_type=args.reward_type)
    env = CausalSetEnv(max_nodes=args.N, proxy=proxy, device=args.device)

    node_feature_dim = 5
    gnn_backbone = GNNPolicy(
        node_feature_dim=node_feature_dim,
        hidden_dim=128,
        out_dim=64
    ).to(args.device)

    policy = FactorPreAggPolicy(
        graph_network=gnn_backbone,
        input_dim=node_feature_dim,
        node_dim=64,
        num_actions=2
    ).to(args.device)

    agent = FactorPreAggAgent(
        policy,
        env,
        collate_fn=collate_fn,
        num_factors_fn=lambda s: s[0],
        device=args.device,
        temperature=args.temp_start,
        epsilon=args.epsilon_start
    )

    gflownet = TBGFlowNet(pf=agent, pb=agent)
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)

    # Training loop
    print(f"Starting training for {args.num_steps} steps...")
    log_data = []
    pbar = tqdm(range(args.num_steps))
    start_time = time.time()

    for step in pbar:
        try:
            # Anneal exploration parameters
            progress = min(step / args.epsilon_decay_steps, 1.0)
            current_epsilon = args.epsilon_start + (args.epsilon_end - args.epsilon_start) * progress
            agent.set_epsilon(current_epsilon)

            progress_temp = min(step / args.temp_decay_steps, 1.0)
            current_temp = args.temp_start + (args.temp_end - args.temp_start) * progress_temp
            agent.set_temperature(current_temp)

            agent.train()
            optimizer.zero_grad()

            # Sample trajectories
            trajectories = gflownet.sample_trajectories(env, n=args.batch_size)

            # Calculate loss
            loss = gflownet.calculate_loss(env, trajectories)

            # Add entropy regularization
            if args.entropy_coef > 0:
                # Get logits from agent
                states_batch = trajectories.states
                logits = agent.get_logits(states_batch)
                probs = torch.softmax(logits, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
                loss = loss - args.entropy_coef * entropy

            if not loss.isfinite():
                print("\nWarning: Non-finite loss detected. Skipping step.")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            if step % 100 == 0 or step == args.num_steps - 1:
                final_states_objects = [t.last_state for t in trajectories]
                final_states = [s.states_list[0] for s in final_states_objects]

                energies_raw = [proxy.get_energy(env.state_to_graph(s)) for s in final_states]
                valid_energies = [e for e in energies_raw if e < 1e9]
                rewards = [math.exp(-args.beta * e) for e in valid_energies]

                avg_energy = np.mean(valid_energies) if valid_energies else 1e9
                avg_reward = np.mean(rewards) if rewards else 0.0

                # Calculate diversity metric
                unique_causets = len(set([tuple(sorted(s[1])) for s in final_states]))
                diversity = unique_causets / len(final_states)

                log_data.append({
                    'step': step,
                    'loss': loss.item(),
                    'avg_energy': avg_energy,
                    'avg_reward': avg_reward,
                    'epsilon': current_epsilon,
                    'temperature': current_temp,
                    'diversity': diversity,
                })

                pbar.set_description(
                    f"Step {step} | Loss: {loss.item():.3f} | Energy: {avg_energy:.3f} | "
                    f"Diversity: {diversity:.2f} | ε: {current_epsilon:.3f}"
                )

        except Exception as e:
            print(f"\n\n--- ERROR AT STEP {step} ---")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            print("Skipping step...")
            if 'optimizer' in locals():
                optimizer.zero_grad()
            time.sleep(1.0)

    print(f"Training complete in {time.time() - start_time:.2f}s")

    # Save logs
    log_df = pd.DataFrame(log_data)
    log_df.to_csv(os.path.join(run_dir, "training_log.csv"), index=False)
    print(f"Saved training log to {run_dir}/training_log.csv")

    # Final sampling (with exploration disabled)
    print("Sampling final ensemble (10,000 samples)...")
    agent.eval()  # This disables epsilon-greedy

    final_graphs = []
    with torch.no_grad():
        while len(final_graphs) < 10000:
            trajectories = gflownet.sample_trajectories(env, n=args.batch_size)
            final_states_objects = [t.last_state for t in trajectories]
            final_states = [s.states_list[0] for s in final_states_objects]
            final_graphs.extend([env.state_to_graph(s) for s in final_states])
            print(f"Collected {len(final_graphs)} / 10000 samples", end='\r')

    ensemble_path = os.path.join(run_dir, "final_ensemble.pkl")
    with open(ensemble_path, 'wb') as f:
        pickle.dump(final_graphs[:10000], f)
    print(f"\nSaved final ensemble to {ensemble_path}")

if __name__ == "__main__":
    main()
