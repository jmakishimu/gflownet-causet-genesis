#!/usr/bin/env python
"""
Simple debug test for staged construction
"""
import torch
from causet_env import CausalSetEnv
from causet_reward import CausalSetRewardProxy


def debug_single_step():
    """Test a single step in detail"""
    print("="*70)
    print("DEBUG: Single Step Test")
    print("="*70)

    proxy = CausalSetRewardProxy(reward_type='bd', device='cpu')
    env = CausalSetEnv(max_nodes=3, proxy=proxy, device='cpu')

    print(f"\nEnvironment: N=3, n_actions={env.n_actions}")
    print(f"Actions: 0=add_node, 1=no_edge, 2=yes_edge, 3=exit")

    # Initial state
    states = env.reset(batch_shape=(1,))
    print(f"\n--- Initial State ---")
    print(f"State tensor: {states.tensor[0, :6]}")
    print(f"n={states.tensor[0, 0].item()}, phase={states.tensor[0, 1].item()}, cursor={states.tensor[0, 2].item()}")
    print(f"Valid actions: {states.forward_masks[0].nonzero().squeeze().tolist()}")

    # Take action 0 (add_node)
    print(f"\n--- Taking action 0 (add_node) ---")
    action = torch.tensor([[0]], dtype=torch.long)
    new_states = env.step(states, action)

    print(f"New state tensor: {new_states.tensor[0, :6]}")
    print(f"n={new_states.tensor[0, 0].item()}, phase={new_states.tensor[0, 1].item()}, cursor={new_states.tensor[0, 2].item()}")
    print(f"Valid actions: {new_states.forward_masks[0].nonzero().squeeze().tolist()}")

    if new_states.tensor[0, 0].item() == 1:
        print("✅ n incremented correctly!")
    else:
        print(f"❌ n did not increment! Expected 1, got {new_states.tensor[0, 0].item()}")
        return False

    # Take more add_node actions
    states = new_states
    for i in range(2):
        print(f"\n--- Taking action 0 (add_node) #{i+2} ---")
        action = torch.tensor([[0]], dtype=torch.long)
        states = env.step(states, action)
        print(f"n={states.tensor[0, 0].item()}, phase={states.tensor[0, 1].item()}, cursor={states.tensor[0, 2].item()}")

    # Should now be in phase 1
    n = states.tensor[0, 0].item()
    phase = states.tensor[0, 1].item()
    cursor = states.tensor[0, 2].item()

    print(f"\n--- After adding all nodes ---")
    print(f"n={n}, phase={phase}, cursor={cursor}")
    print(f"Valid actions: {states.forward_masks[0].nonzero().squeeze().tolist()}")

    if n == 3 and phase == 1 and cursor == 0:
        print("✅ Transitioned to phase 1 correctly!")
    else:
        print(f"❌ Phase transition failed!")
        return False

    # Make edge decision
    print(f"\n--- Taking action 1 (no_edge) ---")
    action = torch.tensor([[1]], dtype=torch.long)
    states = env.step(states, action)

    cursor = states.tensor[0, 2].item()
    print(f"cursor={cursor}")

    if cursor == 1:
        print("✅ Edge cursor advanced!")
        return True
    else:
        print(f"❌ Edge cursor did not advance! Expected 1, got {cursor}")
        return False


if __name__ == "__main__":
    import sys
    success = debug_single_step()
    sys.exit(0 if success else 1)
