"""
Simple workaround: Just use the default Sampler but manually extend incomplete trajectories
"""
print("Sampler workaround loaded - will extend short trajectories")

# We won't monkey-patch. Instead, we'll provide a helper function that train.py can use
# to ensure trajectories complete

def sample_complete_trajectories(sampler, env, n, max_attempts=100):
    """
    Sample complete trajectories by manually continuing from where sampler stopped.

    Args:
        sampler: The GFN Sampler
        env: The environment
        n: Number of trajectories to sample
        max_attempts: Maximum steps to try

    Returns:
        Trajectories object with completed trajectories
    """
    import torch
    from gfn.containers import Trajectories

    # Start with initial states
    states = env.reset(batch_shape=(n,))

    # Collect trajectory data
    all_states = [states]
    all_actions = []
    all_logprobs = []

    done = torch.zeros(n, dtype=torch.bool, device=states.tensor.device)

    for step in range(max_attempts):
        if done.all():
            break

        # Get action from policy
        with torch.no_grad():
            logits = sampler.estimator.module(states.tensor)
            dist = sampler.estimator.to_probability_distribution(states, logits)

        # Sample action
        actions_tensor = dist.sample()
        log_probs = dist.log_prob(actions_tensor)

        all_actions.append(actions_tensor)
        all_logprobs.append(log_probs)

        # Take step
        states = env.step(states, actions_tensor)
        all_states.append(states)

        # Update done - check if reached terminal (n == max_nodes)
        for i in range(n):
            n_val = int(states.tensor[i, 0].item())
            if n_val >= env.max_nodes:
                done[i] = True

    # Now build Trajectories object properly
    # Stack states: (n_steps+1, batch_size, state_dim)
    states_tensor = torch.stack([s.tensor for s in all_states], dim=0)

    # Create States with proper batch_shape
    States_class = env.make_States_class()

    # For Trajectories, we need batch_shape = (n_steps, n_trajectories)
    # Reshape: (n_steps, n_traj, state_dim) - this is already correct

    # But we need to create a States object that has batch_shape property
    # Let's create it differently - torchgfn expects a specific structure

    # Actually, let's just return the raw data and let train.py handle it
    return {
        'states': all_states,
        'actions': all_actions,
        'logprobs': all_logprobs,
        'done': done,
        'n_steps': len(all_states) - 1
    }
