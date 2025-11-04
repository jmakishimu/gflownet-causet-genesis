#!/usr/bin/env python
"""
Manually sample trajectories without using torchgfn's Sampler
to verify our environment works for longer trajectories
"""
import torch
import torch.nn as nn
from causet_env import CausalSetEnv
from causet_reward import CausalSetRewardProxy

class SimplePolicy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.network(x)

def sample_trajectory_manually(env, policy, max_steps=50):
    """Manually sample a trajectory using the policy"""
    states = env.reset(batch_shape=(1,))
    trajectory_states = [states.tensor[0].clone()]
    trajectory_actions = []

    for step in range(max_steps):
        n = int(states.tensor[0, 0].item())

        # Check if terminal
        if n >= env.max_nodes:
            print(f"  Reached terminal state at step {step}")
            break

        # Get policy logits
        with torch.no_grad():
            logits = policy(states.tensor)

        # Apply mask
        masked_logits = logits.clone()
        masked_logits[~states.forward_masks] = float('-inf')

        # Sample action
        probs = torch.softmax(masked_logits, dim=-1)
        action = torch.multinomial(probs[0], 1)

        # Take step
        states = env.step(states, action)

        trajectory_states.append(states.tensor[0].clone())
        trajectory_actions.append(action.item())

        if step % 10 == 0:
            print(f"  Step {step}: n={n}, action={action.item()}")

    return trajectory_states, trajectory_actions

def main():
    print("="*60)
    print("MANUAL TRAJECTORY SAMPLING TEST")
    print("="*60)

    N = 5
    proxy = CausalSetRewardProxy(reward_type='bd')
    env = CausalSetEnv(max_nodes=N, proxy=proxy, device='cpu')

    policy = SimplePolicy(env.state_dim, env.n_actions)

    print(f"\nSampling 3 trajectories manually...")
    print(f"Max steps: 50")
    print()

    completed = 0
    for i in range(3):
        print(f"--- Trajectory {i} ---")
        states, actions = sample_trajectory_manually(env, policy, max_steps=50)

        final_n = int(states[-1][0].item())
        print(f"  Final state: n={final_n}")
        print(f"  Trajectory length: {len(states)} states, {len(actions)} actions")

        if final_n == N:
            print(f"  ✓ COMPLETED")
            completed += 1
        else:
            print(f"  ✗ Incomplete (needed n={N})")
        print()

    print("="*60)
    print(f"Results: {completed}/3 trajectories completed")
    print("="*60)

    if completed > 0:
        print("\n✓ Manual sampling works! The environment can complete trajectories.")
        print("The problem is with torchgfn's Sampler having too short max_length.")
    else:
        print("\n✗ Even manual sampling fails. There may be an issue with the environment.")

if __name__ == "__main__":
    main()
