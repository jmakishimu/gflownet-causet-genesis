#
# gflownet-causet-genesis/train.py
# COMPLETE IMPROVED VERSION with all optimizations
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

from gfn.gflownet.sub_trajectory_balance import SubTBGFlowNet
from gfn.estimators import DiscretePolicyEstimator, ScalarEstimator
from gfn.samplers import Sampler

from causet_env import CausalSetEnv
from causet_reward import CausalSetRewardProxy


class ImprovedCausetPolicy(nn.Module):
    """
    Enhanced policy with:
    - Multi-head self-attention
    - Deeper architecture
    - Layer normalization
    - Dropout for regularization
    """
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int, n_layers: int = 3):
        super().__init__()
        self.input_dim = state_dim

        # Input projection
        self.input_proj = nn.Linear(state_dim, hidden_dim)
        self.input_ln = nn.LayerNorm(hidden_dim)

        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        self.attn_ln = nn.LayerNorm(hidden_dim)

        # Residual blocks
        self.blocks = nn.ModuleList([
            ImprovedResidualBlock(hidden_dim) for _ in range(n_layers)
        ])

        # Output head
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input projection
        h = F.relu(self.input_ln(self.input_proj(x)))

        # Self-attention (helps capture dependencies in state)
        h_seq = h.unsqueeze(1)  # [batch, 1, hidden]
        h_att, _ = self.attention(h_seq, h_seq, h_seq)
        h = self.attn_ln(h + h_att.squeeze(1))  # Residual connection

        # Residual blocks
        for block in self.blocks:
            h = block(h)

        return self.output(h)


class ImprovedResidualBlock(nn.Module):
    """Enhanced residual block with pre-activation and dropout"""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.ln2 = nn.LayerNorm(hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        # Pre-activation
        h = F.relu(self.ln1(x))
        h = self.fc1(h)
        h = F.relu(self.ln2(h))
        h = self.dropout(h)
        h = self.fc2(h)
        return residual + h


class LogFlowModule(nn.Module):
    """Enhanced MLP for estimating log state flows"""
    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = state_dim
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def get_adaptive_beta(step: int, max_steps: int, start: float = 0.1, end: float = 0.1) -> float:
    """
    Adaptive beta schedule: start small, increase over training
    Early: explore broadly with weak penalty
    Late: focus on minimization with strong penalty
    """
    progress = min(step / max_steps, 1.0)
    # Exponential schedule for smoother transition
    return start * (end / start) ** progress


def get_curriculum_N(step: int, max_steps: int, start_N: int = 5, target_N: int = 10) -> int:
    """
    Curriculum learning: start with small graphs, gradually increase
    """
    if step < max_steps * 0.3:
        return start_N
    elif step < max_steps * 0.6:
        return (start_N + target_N) // 2
    else:
        return target_N


def compute_entropy_bonus(states_tensor: torch.Tensor, pf: DiscretePolicyEstimator,
                         masks: torch.Tensor, alpha: float = 0.01) -> torch.Tensor:
    """
    Entropy regularization to encourage exploration
    """
    with torch.no_grad():
        logits = pf.module(states_tensor)
        # Apply mask
        masked_logits = logits.clone()
        masked_logits[~masks] = float('-inf')
        probs = F.softmax(masked_logits, dim=-1)
        # Compute entropy
        entropy = -(probs * torch.log(probs + 1e-8)).sum(-1).mean()
    return alpha * entropy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=10)
    parser.add_argument("--reward_type", type=str, default="bd", choices=["bd", "mmd"])

    # Adaptive beta schedule
    parser.add_argument("--beta_start", type=float, default=0.1)
    parser.add_argument("--beta_end", type=float, default=0.1)

    # Training
    parser.add_argument("--num_steps", type=int, default=20_000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=4)

    # Curriculum learning
    parser.add_argument("--use_curriculum", action="store_true")
    parser.add_argument("--start_N", type=int, default=5)

    # Optimization
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--entropy_coef", type=float, default=0.01)

    # Device
    parser.add_argument("--device", type=str,
                       default="cuda" if torch.cuda.is_available() else "cpu")

    # Exploration
    parser.add_argument("--epsilon_start", type=float, default=0.3)
    parser.add_argument("--epsilon_end", type=float, default=0.0)
    parser.add_argument("--epsilon_decay_steps", type=int, default=10000)
    parser.add_argument("--temp_start", type=float, default=2.0)
    parser.add_argument("--temp_end", type=float, default=1.0)
    parser.add_argument("--temp_decay_steps", type=int, default=10000)

    # Saving
    parser.add_argument("--output_dir", type=str, default="experiment_results")
    parser.add_argument("--run_name", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.run_name is None:
        curriculum_tag = "curr" if args.use_curriculum else "nocurr"
        args.run_name = (f"improved_n{args.N}_{args.reward_type}_"
                        f"b{args.batch_size}_h{args.hidden_dim}_{curriculum_tag}")

    run_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    print("="*70)
    print("IMPROVED CAUSET GFLOWNET TRAINING")
    print("="*70)
    print(f"Run: {args.run_name}")
    print(f"Device: {args.device}")
    print(f"Target N: {args.N}, Batch: {args.batch_size}, Steps: {args.num_steps}")
    print(f"Improvements:")
    print(f"  - Adaptive beta: {args.beta_start:.4f} → {args.beta_end:.4f}")
    print(f"  - Enhanced architecture: attention + deeper residual blocks")
    print(f"  - Entropy regularization: coef={args.entropy_coef}")
    if args.use_curriculum:
        print(f"  - Curriculum learning: N={args.start_N} → {args.N}")
    print("="*70 + "\n")

    # Initialize environment
    print("Initializing environment...")
    proxy = CausalSetRewardProxy(reward_type=args.reward_type, device=args.device)

    # Start with curriculum or target N
    current_N = args.start_N if args.use_curriculum else args.N
    env = CausalSetEnv(max_nodes=current_N, proxy=proxy, device=args.device)

    print(f"Initial N: {current_N}")
    print(f"State dim: {env.state_dim}, Action dim: {env.n_actions}")

    # Create forward policy with improved architecture
    print("Creating improved policy network...")
    pf_module = ImprovedCausetPolicy(
        state_dim=env.state_dim,
        hidden_dim=args.hidden_dim,
        action_dim=env.n_actions,
        n_layers=args.n_layers
    )

    pf = DiscretePolicyEstimator(
        module=pf_module,
        n_actions=env.n_actions,
        preprocessor=None,
        is_backward=False
    )

    pf.to(args.device)

    # Create log flow estimator
    logF_module = LogFlowModule(env.state_dim, args.hidden_dim)
    logF = ScalarEstimator(module=logF_module, preprocessor=None)
    logF.to(args.device)

    n_params_pf = sum(p.numel() for p in pf.parameters())
    n_params_logF = sum(p.numel() for p in logF.parameters())
    print(f"PF parameters: {n_params_pf:,}")
    print(f"LogF parameters: {n_params_logF:,}")
    print(f"Total: {n_params_pf + n_params_logF:,}\n")

    # Initialize GFlowNet with SubTB
    print("Initializing SubTB GFlowNet...")
    gflownet = SubTBGFlowNet(
        pf=pf,
        pb=None,
        logF=logF,
        weighting="geometric_within",
        lamda=0.9,
        constant_pb=True
    )

    # Optimizer with weight decay
    optimizer = optim.AdamW(
        list(pf.parameters()) + list(logF.parameters()),
        lr=args.lr,
        weight_decay=1e-4
    )

    # Learning rate scheduler with warmup
    def lr_lambda(step):
        warmup_steps = 1000
        if step < warmup_steps:
            return step / warmup_steps
        else:
            # Cosine decay after warmup
            progress = (step - warmup_steps) / (args.num_steps - warmup_steps)
            return 0.1 + 0.9 * (1 + np.cos(np.pi * progress)) / 2

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Create sampler
    sampler = Sampler(estimator=pf)

    # Training loop
    print("Starting training...\n")
    log_data = []
    start_time = time.time()

    pbar = tqdm(range(args.num_steps), desc="Training")

    for step in pbar:
        # === ADAPTIVE BETA SCHEDULE ===
        current_beta = get_adaptive_beta(
            step, args.num_steps,
            args.beta_start, args.beta_end
        )
        proxy.beta = current_beta  # Update proxy's beta

        # === CURRICULUM LEARNING ===
        if args.use_curriculum:
            new_N = get_curriculum_N(step, args.num_steps, args.start_N, args.N)
            if new_N != current_N and step % 100 == 0:
                current_N = new_N
                print(f"\n[Step {step}] Curriculum: Increasing N to {current_N}")
                env = CausalSetEnv(max_nodes=current_N, proxy=proxy, device=args.device)
                # Note: policy will handle variable state dimensions via padding

        # === EXPLORATION ANNEALING ===
        progress = min(step / args.epsilon_decay_steps, 1.0)
        current_epsilon = args.epsilon_start + (args.epsilon_end - args.epsilon_start) * progress

        progress_temp = min(step / args.temp_decay_steps, 1.0)
        current_temp = args.temp_start + (args.temp_end - args.temp_start) * progress_temp

        # === TRAINING STEP ===
        pf.train()
        logF.train()
        optimizer.zero_grad()

        # Sample trajectories
        trajectories = sampler.sample_trajectories(
            env=env,
            n=args.batch_size,
            temperature=current_temp,
            epsilon=current_epsilon
        )

        # Compute SubTB loss
        loss = gflownet.loss(env, trajectories)

        if not loss.isfinite():
            pbar.write(f"Warning: Non-finite loss at step {step}")
            continue

        # === ENTROPY REGULARIZATION ===
        # Compute entropy bonus on sampled states
        all_states = trajectories.states.tensor
        if all_states.dim() == 3:
            # Flatten to [n_steps * batch, state_dim]
            states_flat = all_states.reshape(-1, all_states.shape[-1])
            # Filter out sink states
            non_sink_mask = states_flat[:, 0] != -1
            if non_sink_mask.any():
                valid_states = states_flat[non_sink_mask]
                # Create dummy states object for mask computation
                temp_states = env.States(valid_states[:args.batch_size])  # Sample subset
                entropy = compute_entropy_bonus(
                    temp_states.tensor, pf,
                    temp_states.forward_masks,
                    alpha=args.entropy_coef
                )
                # Add entropy bonus (negative because we maximize entropy)
                total_loss = loss - entropy
            else:
                total_loss = loss
        else:
            total_loss = loss

        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(pf.parameters()) + list(logF.parameters()),
            args.grad_clip
        )
        optimizer.step()
        scheduler.step()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.2e}',
            'beta': f'{current_beta:.4f}',
            'N': current_N,
            'lr': f'{scheduler.get_last_lr()[0]:.2e}',
            'eps': f'{current_epsilon:.3f}'
        })

        # === EVALUATION ===
        if step % args.eval_every == 0 or step == args.num_steps - 1:
            with torch.no_grad():
                all_states = trajectories.states.tensor

                if all_states.dim() == 3:
                    batch_size_actual = all_states.shape[1]

                    # Extract final states
                    final_states_list = []
                    for b in range(batch_size_actual):
                        traj = all_states[:, b, :]
                        non_sink = traj[:, 0] != -1
                        if non_sink.any():
                            final_states_list.append(traj[non_sink][-1])

                    if final_states_list:
                        final_states_batch = torch.stack(final_states_list)

                        # BATCHED energy computation
                        if args.reward_type == 'bd':
                            energies = proxy.get_bd_energy_batched(
                                final_states_batch,
                                current_N
                            ).cpu().numpy()
                        else:
                            energies = []
                            for state in final_states_batch:
                                g = env._state_to_graph(state)
                                energies.append(proxy.get_energy(g))
                            energies = np.array(energies)

                        # Filter valid energies
                        valid_energies = energies[energies < 1e9]

                        avg_energy = np.mean(valid_energies) if len(valid_energies) > 0 else 1e9
                        min_energy = np.min(valid_energies) if len(valid_energies) > 0 else 1e9
                        avg_log_reward = -current_beta * avg_energy**2 if len(valid_energies) > 0 else -1e9
                        valid_frac = len(valid_energies) / batch_size_actual

                        # Calculate diversity
                        unique_graphs = len(set(tuple(s.cpu().numpy()) for s in final_states_batch))
                        diversity = unique_graphs / len(final_states_batch)
                    else:
                        avg_energy = 1e9
                        min_energy = 1e9
                        avg_log_reward = -1e9
                        valid_frac = 0.0
                        diversity = 0.0

                    log_data.append({
                        'step': step,
                        'loss': loss.item(),
                        'avg_energy': avg_energy,
                        'min_energy': min_energy,
                        'avg_log_reward': avg_log_reward,
                        'valid_fraction': valid_frac,
                        'diversity': diversity,
                        'epsilon': current_epsilon,
                        'temperature': current_temp,
                        'beta': current_beta,
                        'current_N': current_N,
                        'lr': scheduler.get_last_lr()[0],
                        'elapsed_time': time.time() - start_time
                    })

                    if step % (args.eval_every * 5) == 0:
                        pbar.write(
                            f"Step {step}: loss={loss.item():.2e}, "
                            f"E_avg={avg_energy:.2f}, E_min={min_energy:.2f}, "
                            f"logR={avg_log_reward:.2f}, valid={valid_frac:.2%}, "
                            f"div={diversity:.2%}, β={current_beta:.4f}, N={current_N}"
                        )

    pbar.close()

    # === FINAL EVALUATION WITH TARGET N ===
    if args.use_curriculum and current_N != args.N:
        print(f"\nSwitching to target N={args.N} for final evaluation...")
        env = CausalSetEnv(max_nodes=args.N, proxy=proxy, device=args.device)
        current_N = args.N

    # Save results
    print("\nSaving results...")
    df = pd.DataFrame(log_data)
    df.to_csv(os.path.join(run_dir, "training_log.csv"), index=False)

    torch.save({
        'pf': pf.state_dict(),
        'logF': logF.state_dict(),
        'optimizer': optimizer.state_dict(),
        'args': vars(args)
    }, os.path.join(run_dir, "checkpoint.pt"))

    # Generate final ensemble at target N with strong beta
    print(f"Generating final ensemble (1000 samples at N={args.N})...")
    proxy.beta = args.beta_end  # Use strongest penalty
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

    # Calculate final statistics
    if final_ensemble:
        final_energies = [proxy.get_bd_energy(g) for g in final_ensemble[:100]]
        mean_energy = np.mean(final_energies)
        min_energy = np.min(final_energies)
        std_energy = np.std(final_energies)

        print(f"\nFinal Ensemble Statistics (N={args.N}):")
        print(f"  Size: {len(final_ensemble)} graphs")
        print(f"  Mean S_BD: {mean_energy:.2f} ± {std_energy:.2f}")
        print(f"  Min S_BD: {min_energy:.2f}")
        print(f"  Target: 0.0")
        print(f"  Improvement from random: TBD (compare with baseline)")

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
