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

# --- UPDATED IMPORTS ---
from gfn.gflownet.trajectory_balance import TBGFlowNet
# The warning told us to import from 'estimators' instead of 'modules'
# NOTE: This class is MISSING from the library files you provided.
from custom_gfn import FactorPreAggAgent
from custom_gfn import FactorPreAggPolicy
# ---
# This import was also incorrect. GFlowNetTrainer is in gfn.utils.training
from torch_geometric.data import Data, Batch
from torch_geometric.utils import from_networkx

# Import our custom modules
from causet_env import CausalSetEnv
from causet_policy import GNNPolicy
from causet_reward import CausalSetRewardProxy

# --- DEBUG ---
_collate_fn_call_count = 0
_last_collate_debug_print = 0
# -----------

def collate_fn(states: list[tuple]) -> Batch:
    """
    (Fix #4) Collates environment states into a PyG Batch
    with rich node features.
    """
    # --- DEBUG ---
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
    # -----------

    data_list = []
    for (n, edges, _) in states:
        if n == 0:
            # Empty graph
            data = Data(
                x=torch.empty(0, 5).float(),
                edge_index=torch.empty(2, 0).long(),
                num_nodes=0
            )
        else:
            g = nx.DiGraph()
            g.add_nodes_from(range(n))
            g.add_edges_from(edges)

            # (Fix #3 Applied) Create rich node features efficiently.
            try:
                # Use reflexive=False for ancestor/descendant counts
                # Use reflexive=True for link prediction (not needed here)
                tc = nx.transitive_closure_dag(g) # More efficient for DAGs
            except Exception:
                # Fallback for safety, though causets should be DAGs
                tc = nx.transitive_closure(g, reflexive=False)


            features = []
            for i in range(n):
                feat = [
                    float(i), # Node index (time of birth)
                    float(g.in_degree(i)),  # Direct parents
                    float(g.out_degree(i)), # Direct children
                    float(tc.in_degree(i)), # Total ancestors
                    float(tc.out_degree(i)) # Total descendants
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

    # --- Setup All Components ---

    # 2. Reward Proxy
    proxy = CausalSetRewardProxy(reward_type=args.reward_type)

    # 1. Environment (FactorEnv)
    # --- UPDATED ---
    # The proxy is now passed to the environment
    env = CausalSetEnv(max_nodes=args.N, proxy=proxy, device=args.device)
    # ---

    # 3. GNN Backbone
    # (Define node_feature_dim explicitly)
    node_feature_dim = 5
    gnn_backbone = GNNPolicy(
        node_feature_dim=node_feature_dim,
        hidden_dim=128,
        out_dim=64
    ).to(args.device)

    # 4. GFlowNet Factored Policy
    policy = FactorPreAggPolicy(
        graph_network=gnn_backbone,
        input_dim=node_feature_dim, # <-- FIX: Pass input_dim
        node_dim=64,                # GNN output dim
        num_actions=2               # (0, 1)
    ).to(args.device)

    # 5. GFlowNet Agent
    agent = FactorPreAggAgent(
        policy,
        env,
        collate_fn=collate_fn,
        num_factors_fn=lambda s: s[0],
        device=args.device
    )

    # 6. GFlowNet Algorithm (Trajectory Balance)
    # The new class takes pf (forward policy) and pb (backward policy).
    # The 'agent' object serves as both.
    gflownet = TBGFlowNet(pf=agent, pb=agent)

    # 7. Optimizer
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)

    # --- Run Training (Phase 4) ---
    print(f"Starting training for {args.num_steps} steps...")

    log_data = []
    pbar = tqdm(range(args.num_steps))
    start_time = time.time()

    for step in pbar:
        try:
            agent.policy.train()
            optimizer.zero_grad()

            # Sample trajectories
            # The 'env' must be passed to the sampling method.
            # ---
            # --- THE FIX ---
            # The library's sampler (gfn/samplers.py) uses the 'n'
            # argument as the BATCH SIZE and the MAX_LEN.
            # We must pass args.batch_size here, not args.N,
            # to ensure we sample the correct number of trajectories.
            #
            # --- BUG #2 FIX: Corrected keyword argument ---
            # The API uses 'n', not 'n_samples'.
            trajectories = gflownet.sample_trajectories(
                env, n=args.batch_size
            )
            # -----------

            # Calculate loss
            # The 'env' must also be passed to the loss calculation.
            loss = gflownet.calculate_loss(env, trajectories)

            if not loss.isfinite():
                print("\nWarning: Non-finite loss detected. Skipping step.")
                # --- DEBUG ---
                print("Dumping first trajectory from bad batch:")
                print(trajectories[0])
                # -----------
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            if step % 100 == 0 or step == args.num_steps - 1:
                # Log data
                final_states_objects = [t.last_state for t in trajectories]
                # --- DEBUG ---
                # The old code `[t[-1]['state'] for t in trajectories]` is wrong
                # for the 'torchgfn' library. Trajectories are objects.
                # Use .states_list[0] to get the raw tuple from our ObjectStates
                final_states = [s.states_list[0] for s in final_states_objects]
                # -----------

                # This logging code is still valid
                energies_raw = [proxy.get_energy(env.state_to_graph(s)) for s in final_states]

                # Filter invalid energies
                valid_energies = [e for e in energies_raw if e < 1e9]
                rewards = [math.exp(-args.beta * e) for e in valid_energies]

                avg_energy = np.mean(valid_energies) if valid_energies else 1e9
                avg_reward = np.mean(rewards) if rewards else 0.0

                log_data.append({
                    'step': step,
                    'loss': loss.item(),
                    'avg_energy': avg_energy,
                    'avg_reward': avg_reward,
                })

                pbar.set_description(f"Step {step} | Loss: {loss.item():.3f} | Avg. Energy: {avg_energy:.3f}")

        # --- DEBUG ---
        except Exception as e:
            print(f"\n\n--- CRITICAL ERROR AT STEP {step} ---")
            print(f"Error: {e}")
            print("--- Traceback ---")
            import traceback
            traceback.print_exc()
            print("-----------------")
            print("Skipping step and attempting to continue...")
            if 'optimizer' in locals():
                optimizer.zero_grad() # Clear grads to avoid corruption
            time.sleep(1.0) # Pause to allow user to see error
        # -----------


    print(f"Training complete in {time.time() - start_time:.2f}s")

    # --- Save Training Logs ---
    log_df = pd.DataFrame(log_data)
    log_df.to_csv(os.path.join(run_dir, "training_log.csv"), index=False)
    print(f"Saved training log to {run_dir}/training_log.csv")

    # --- Run Sampling (Phase 4) ---
    print("Sampling final ensemble (10,000 samples)...")
    agent.policy.eval()

    final_graphs = []
    with torch.no_grad():
        while len(final_graphs) < 10000:
            # Also need to pass 'env' here for the final sampling.
            # ---
            # --- THE FIX ---
            # Using args.batch_size for 'n' here as well.
            # --- BUG #2 FIX: Corrected keyword argument ---
            trajectories = gflownet.sample_trajectories(
                env, n=args.batch_size
            )
            # -----------
            # --- DEBUG ---
            # Correctly extract final states from trajectory objects
            final_states_objects = [t.last_state for t in trajectories]
            final_states = [s.states_list[0] for s in final_states_objects]
            final_graphs.extend([
                env.state_to_graph(s) for s in final_states
            ])
            # -----------
            print(f"Collected {len(final_graphs)} / 10000 samples", end='\r')

    # Save ensemble
    ensemble_path = os.path.join(run_dir, "final_ensemble.pkl")
    # --- THIS LINE IS FIXED ---
    with open(ensemble_path, 'wb') as f:
        pickle.dump(final_graphs[:10000], f)
    print(f"\nSaved final ensemble to {ensemble_path}")

if __name__ == "__main__":
    main()
