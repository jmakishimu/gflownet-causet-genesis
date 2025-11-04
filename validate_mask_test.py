#!/usr/bin/env python
"""
Quick diagnostic test to verify the fix works
"""
import torch
import torch.nn as nn
from causet_env import CausalSetEnv
from causet_reward import CausalSetRewardProxy
from gfn.estimators import DiscretePolicyEstimator
from gfn.samplers import Sampler
from gfn.gflownet.trajectory_balance import TBGFlowNet


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


def main():
    print("="*70)
    print("QUICK DIAGNOSTIC TEST")
    print("="*70)

    # Setup
    proxy = CausalSetRewardProxy(reward_type='bd')
    env = CausalSetEnv(max_nodes=5, proxy=proxy, device='cpu')

    print(f"\nEnvironment: N={env.max_nodes}, n_actions={env.n_actions}")

    # Test 1: Check masks at sink state
    print("\n--- TEST 1: Sink State Masks ---")
    sf_state = env.States(env.sf.unsqueeze(0))
    print(f"s_f forward masks: {sf_state.forward_masks[0].sum().item()} valid actions")
    print(f"s_f backward masks: {sf_state.backward_masks[0].sum().item()} valid actions")

    if sf_state.forward_masks[0].sum().item() == 0:
        print("❌ FAIL: s_f has no valid forward actions!")
        return False

    if sf_state.backward_masks[0].sum().item() == 0:
        print("❌ FAIL: s_f has no valid backward actions!")
        return False

    print("✓ s_f has valid masks")

    # Test 2: Create policies
    print("\n--- TEST 2: Policy Creation ---")
    pf_module = SimplePolicy(env.state_dim, env.n_actions)
    pf = DiscretePolicyEstimator(
        module=pf_module,
        n_actions=env.n_actions,
        is_backward=False,
        preprocessor=None
    )

    pb_module = SimplePolicy(env.state_dim, env.n_actions - 1)
    pb = DiscretePolicyEstimator(
        module=pb_module,
        n_actions=env.n_actions,
        is_backward=True,
        preprocessor=None
    )

    print(f"✓ pf created (output_dim={env.n_actions})")
    print(f"✓ pb created (output_dim={env.n_actions - 1})")

    # Test 3: Check pb at sink state
    print("\n--- TEST 3: Backward Policy at Sink State ---")
    with torch.no_grad():
        logits_pb = pb.module(sf_state.tensor)
        print(f"pb logits shape: {logits_pb.shape}")
        print(f"sf_state.tensor shape: {sf_state.tensor.shape}")
        print(f"sf_state.batch_shape: {sf_state.batch_shape}")

        dist_pb = pb.to_probability_distribution(sf_state, logits_pb)
        print(f"Distribution created successfully")

        # Sample an action to test
        sampled_action = dist_pb.sample()
        print(f"Sampled action shape: {sampled_action.shape}")
        print(f"Sampled action: {sampled_action}")

        # Compute log prob for the sampled action
        log_prob = dist_pb.log_prob(sampled_action)

        print(f"pb log_prob at s_f: {log_prob.item():.4f}")

        if not torch.isfinite(log_prob):
            print("❌ FAIL: pb gives -inf log prob at s_f!")
            return False

        print("✓ pb can compute finite log probs at s_f")

    # Test 4: Sample trajectories
    print("\n--- TEST 4: Trajectory Sampling ---")
    gflownet = TBGFlowNet(pf=pf, pb=pb)
    sampler = Sampler(estimator=pf)

    try:
        trajectories = sampler.sample_trajectories(
            env=env,
            n=4,
            temperature=1.0,
            epsilon=0.1
        )
        print(f"✓ Sampled trajectories: shape={trajectories.states.tensor.shape}")
    except Exception as e:
        print(f"❌ FAIL: Sampling failed: {e}")
        return False

    # Test 5: Compute loss
    print("\n--- TEST 5: TB Loss Computation ---")
    try:
        loss = gflownet.loss(env, trajectories)
        print(f"Loss: {loss.item():.4e}")

        if torch.isfinite(loss):
            print("✅ SUCCESS: Loss is finite!")
            return True
        else:
            print("❌ FAIL: Loss is -inf or nan")
            return False

    except Exception as e:
        print(f"❌ FAIL: Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()

    print("\n" + "="*70)
    if success:
        print("🎉 ALL TESTS PASSED - Ready for training!")
        print("\nRun: python train_fixed.py --N 5 --num_steps 100 --batch_size 4")
    else:
        print("⚠️  TESTS FAILED - Review output above")
    print("="*70)

    import sys
    sys.exit(0 if success else 1)
