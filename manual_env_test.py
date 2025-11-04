#!/usr/bin/env python
"""
Manual test to verify the environment works correctly
"""
import torch
from causet_env import CausalSetEnv
from causet_reward import CausalSetRewardProxy

def test_manual_trajectory():
    """Test a manual trajectory to see if we can reach terminal state"""
    print("Testing manual trajectory...")

    proxy = CausalSetRewardProxy(reward_type='bd')
    env = CausalSetEnv(max_nodes=5, proxy=proxy, device='cpu')

    print(f"Environment initialized:")
    print(f"  max_nodes: {env.max_nodes}")
    print(f"  n_actions: {env.n_actions}")
    print(f"  state_dim: {env.state_dim}")

    # Reset environment
    states = env.reset(batch_shape=(1,))
    print(f"\nInitial state:")
    print(f"  n = {int(states.tensor[0, 0].item())}")
    print(f"  forward_masks shape: {states.forward_masks.shape}")
    print(f"  valid actions: {states.forward_masks[0].nonzero().squeeze()}")

    # Take actions to build a trajectory
    step = 0
    while int(states.tensor[0, 0].item()) < env.max_nodes:
        n = int(states.tensor[0, 0].item())
        print(f"\n--- Step {step}, n={n} ---")

        # Get valid actions
        valid_actions = states.forward_masks[0].nonzero().squeeze()
        print(f"Valid actions: {valid_actions}")

        # Take the exit action (last valid action, which should be n_actions-1)
        action = env.n_actions - 1
        print(f"Taking action: {action} (exit action)")

        # Create action tensor
        action_tensor = torch.tensor([[action]], dtype=torch.long)

        # Step
        states = env.step(states, action_tensor)

        new_n = int(states.tensor[0, 0].item())
        print(f"New n: {new_n}")

        step += 1
        if step > 10:  # Safety limit
            print("ERROR: Too many steps!")
            break

    print(f"\n=== Final State ===")
    print(f"n = {int(states.tensor[0, 0].item())}")
    print(f"Is terminal: {int(states.tensor[0, 0].item()) == env.max_nodes}")

    if int(states.tensor[0, 0].item()) == env.max_nodes:
        # Calculate reward
        final_state = states.tensor[0]
        g = env._state_to_graph(final_state)
        energy = proxy.get_energy(g)
        log_reward = -energy
        print(f"Energy: {energy}")
        print(f"Log reward: {log_reward}")
        print(f"Graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
        print("SUCCESS: Reached terminal state!")
    else:
        print("FAILURE: Did not reach terminal state")

if __name__ == "__main__":
    test_manual_trajectory()
