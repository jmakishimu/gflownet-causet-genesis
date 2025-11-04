#
# gflownet-causet-genesis/train.py
# OPTIMIZED VERSION - Fixed warnings, faster training
#
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import pickle
import argparse
import os
from tqdm import tqdm
import time

from gfn.gflownet.trajectory_balance import TBGFlowNet
from gfn.estimators import DiscretePolicyEstimator
from gfn.samplers import Sampler

from causet_env import CausalSetEnv
from causet_reward import CausalSetRewardProxy


class OptimizedCausetPolicy(nn.Module):
    """Optimized policy network with layer norm and residual connections"""
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int, n_layers: int = 3):
        super().__init__()
        self.input_dim = state_dim

        self.input_proj = nn.Linear(state_dim, hidden_dim)
        self.input_ln = nn.LayerNorm(hidden_dim)

        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(n_layers)
        ])

        self.output = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.input_ln(self.input_proj(x)))

        for block in self.blocks:
            x = block(x)

        return self.output(x)


class ResidualBlock(nn.Module):
    """Residual block with layer norm"""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.ln2(self.fc2(x))
        return F.relu(x + residual)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=10)
    parser.add_argument("--reward_type", type=str, default="bd", choices=["bd", "mmd"])
    parser.add_argument("--beta", type=float, default=1.0)

    # Training
    parser.add_argument("--num_steps", type=int, default=10_000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=3)

    # Optimization
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eval_every", type=int, default=100)

    # Device
    parser.add_argument("--device", type=str,
                       default="cuda" if torch.cuda.is_available() else "cpu")

    # Exploration
    parser.add_argument("--epsilon_start", type=float, default=0.3)
    parser.add_argument("--epsilon_end", type=float, default=0.05)
    parser.add_argument("--epsilon_decay_steps", type=int, default=5000)
    parser.add_argument("--temp_start", type=float, default=2.0)
    parser.add_argument("--temp_end", type=float, default=1.0)
    parser.add_argument("--temp_decay_steps", type=int, default=5000)

    # Saving
    parser.add_argument("--output_dir", type=str, default="experiment_results")
    parser.add_argument("--run_name", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.run_name is None:
        args.run_name = f"n{args.N}_{args.reward_type}_b{args.batch_size}_h{args.hidden_dim}"

    run_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    print("="*70)
    print("OPTIMIZED CAUSET GFLOWNET TRAINING")
    print("="*70)
    print(f"Run: {args.run_name}")
    print(f"Device: {args.device}")
    print(f"N: {args.N}, Batch: {args.batch_size}, Steps: {args.num_steps}")
    print(f"AMP: {args.use_amp}")
    print("="*70 + "\n")

    # Initialize environment
    print("Initializing environment...")
    proxy = CausalSetRewardProxy(reward_type=args.reward_type, device=args.device)
    env = CausalSetEnv(max_nodes=args.N, proxy=proxy, device=args.device)

    print(f"State dim: {env.state_dim}, Action dim: {env.n_actions}")

    # Create optimized policies
    print("Creating policy networks...")
    pf_module = OptimizedCausetPolicy(
        state_dim=env.state_dim,
        hidden_dim=args.hidden_dim,
        action_dim=env.n_actions,
        n_layers=args.n_layers
    )

    pb_module = OptimizedCausetPolicy(
        state_dim=env.state_dim,
        hidden_dim=args.hidden_dim,
        action_dim=env.n_actions - 1,
        n_layers=args.n_layers
    )

    pf = DiscretePolicyEstimator(
        module=pf_module,
        n_actions=env.n_actions,
        preprocessor=None,
        is_backward=False
    )

    pb = DiscretePolicyEstimator(
        module=pb_module,
        n_actions=env.n_actions,
        preprocessor=None,
        is_backward=True
    )

    pf.to(args.device)
    pb.to(args.device)

    n_params_pf = sum(p.numel() for p in pf.parameters())
    n_params_pb = sum(p.numel() for p in pb.parameters())
    print(f"PF parameters: {n_params_pf:,}")
    print(f"PB parameters: {n_params_pb:,}")
    print(f"Total: {n_params_pf + n_params_pb:,}\n")

    # Initialize GFlowNet
    gflownet = TBGFlowNet(pf=pf, pb=pb)

    # Optimizer with weight decay
    optimizer = optim.AdamW(
        list(pf.parameters()) + list(pb.parameters()),
        lr=args.lr,
        weight_decay=1e-4
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_steps, eta_min=args.lr * 0.1
    )

    # FIXED: Use torch.amp instead of torch.cuda.amp
    scaler = torch.amp.GradScaler('cuda') if args.use_amp and args.device == 'cuda' else None

    # Create sampler
    sampler = Sampler(estimator=pf)

    # Training loop
    print("Starting training...\n")
    log_data = []
    start_time = time.time()

    pbar = tqdm(range(args.num_steps), desc="Training")

    for step in pbar:
        # Anneal exploration
        progress = min(step / args.epsilon_decay_steps, 1.0)
        current_epsilon = args.epsilon_start + (args.epsilon_end - args.epsilon_start) * progress

        progress_temp = min(step / args.temp_decay_steps, 1.0)
        current_temp = args.temp_start + (args.temp_end - args.temp_start) * progress_temp

        # Training step
        pf.train()
        pb.train()
        optimizer.zero_grad()

        # FIXED: Use torch.amp.autocast instead of torch.cuda.amp.autocast
        with torch.amp.autocast('cuda', enabled=args.use_amp and args.device == 'cuda'):
            trajectories = sampler.sample_trajectories(
                env=env,
                n=args.batch_size,
                temperature=current_temp,
                epsilon=current_epsilon
            )

            loss = gflownet.loss(env, trajectories)

        if not loss.isfinite():
            pbar.write(f"Warning: Non-finite loss at step {step}")
            continue

        # Backward pass with optional AMP
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(pf.parameters(), args.grad_clip)
            torch.nn.utils.clip_grad_norm_(pb.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pf.parameters(), args.grad_clip)
            torch.nn.utils.clip_grad_norm_(pb.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()


        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.2e}',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}'
        })

        # OPTIMIZED EVALUATION
        if step % args.eval_every == 0 or step == args.num_steps - 1:
            with torch.no_grad():
                all_states = trajectories.states.tensor

                if all_states.dim() == 3:
                    batch_size_actual = all_states.shape[1]

                    # Extract final states more efficiently
                    final_states_list = []
                    for b in range(batch_size_actual):
                        traj = all_states[:, b, :]
                        non_sink = traj[:, 0] != -1
                        if non_sink.any():
                            final_states_list.append(traj[non_sink][-1])

                    if final_states_list:
                        # Stack all final states
                        final_states_batch = torch.stack(final_states_list)

                        # BATCHED energy computation (MUCH FASTER)
                        if args.reward_type == 'bd':
                            energies = proxy.get_bd_energy_batched(
                                final_states_batch,
                                args.N
                            ).cpu().numpy()
                        else:
                            # Sequential for MMD (ablation)
                            energies = []
                            for state in final_states_batch:
                                g = env._state_to_graph(state)
                                energies.append(proxy.get_energy(g))
                            energies = np.array(energies)

                        # Filter valid energies
                        valid_energies = energies[energies < 1e9]

                        avg_energy = np.mean(valid_energies) if len(valid_energies) > 0 else 1e9
                        avg_log_reward = -args.beta * avg_energy if len(valid_energies) > 0 else -1e9
                        valid_frac = len(valid_energies) / batch_size_actual
                    else:
                        avg_energy = 1e9
                        avg_log_reward = -1e9
                        valid_frac = 0.0

                    log_data.append({
                        'step': step,
                        'loss': loss.item(),
                        'avg_energy': avg_energy,
                        'avg_log_reward': avg_log_reward,
                        'valid_fraction': valid_frac,
                        'epsilon': current_epsilon,
                        'temperature': current_temp,
                        'lr': scheduler.get_last_lr()[0],
                        'elapsed_time': time.time() - start_time
                    })

                    if step % (args.eval_every * 5) == 0:
                        pbar.write(
                            f"Step {step}: loss={loss.item():.2e}, "
                            f"E={avg_energy:.2e}, logR={avg_log_reward:.2f}, "
                            f"valid={valid_frac:.2%}"
                        )

    pbar.close()

    # Save results
    print("\nSaving results...")
    df = pd.DataFrame(log_data)
    df.to_csv(os.path.join(run_dir, "training_log.csv"), index=False)

    torch.save({
        'pf': pf.state_dict(),
        'pb': pb.state_dict(),
        'optimizer': optimizer.state_dict(),
        'args': vars(args)
    }, os.path.join(run_dir, "checkpoint.pt"))

    # Generate final ensemble
    print("Generating final ensemble (1000 samples)...")
    pf.eval()
    final_ensemble = []

    with torch.no_grad():
        n_batches = 1000 // args.batch_size + 1
        for _ in tqdm(range(n_batches), desc="Sampling"):
            trajectories = sampler.sample_trajectories(
                env=env,
                n=args.batch_size,
                temperature=1.0,
                epsilon=0.0
            )

            all_states = trajectories.states.tensor
            if all_states.dim() == 3:
                for b in range(all_states.shape[1]):
                    traj = all_states[:, b, :]
                    non_sink = traj[:, 0] != -1
                    if non_sink.any():
                        final_state = traj[non_sink][-1]
                        if int(final_state[0].item()) == args.N:
                            g = env._state_to_graph(final_state)
                            final_ensemble.append(g)

    final_ensemble = final_ensemble[:1000]

    with open(os.path.join(run_dir, "final_ensemble.pkl"), 'wb') as f:
        pickle.dump(final_ensemble, f)

    print(f"\n{'='*70}")
    print(f"Training complete!")
    print(f"Time: {(time.time() - start_time) / 60:.1f} minutes")
    print(f"Final ensemble: {len(final_ensemble)} graphs")
    print(f"Results saved to: {run_dir}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
