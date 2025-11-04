#!/usr/bin/env python
"""Verify that we're using the optimized batched code"""
import torch
from causet_env import CausalSetEnv
from causet_reward import CausalSetRewardProxy

def verify_setup():
    print("="*70)
    print("VERIFICATION: Checking optimized code is loaded")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Check proxy
    print("\n1. Checking CausalSetRewardProxy...")
    proxy = CausalSetRewardProxy(reward_type='bd', device=device)

    if hasattr(proxy, 'get_bd_energy_batched'):
        print("   ✓ Batched BD energy method found")
    else:
        print("   ✗ ERROR: Batched method missing!")
        return False

    # Check environment
    print("\n2. Checking CausalSetEnv...")
    env = CausalSetEnv(max_nodes=5, proxy=proxy, device=device)

    # Test batched computation
    print("\n3. Testing batched computation...")
    batch_size = 4
    test_states = env.s0.repeat(batch_size, 1)
    test_states[:, 0] = 5  # Set n=5 for all

    try:
        energies = proxy.get_bd_energy_batched(test_states, 5)
        print(f"   ✓ Batched computation works! Shape: {energies.shape}")
        print(f"   Sample energies: {energies[:3]}")
    except Exception as e:
        print(f"   ✗ ERROR: {e}")
        return False

    print("\n" + "="*70)
    print("✅ ALL CHECKS PASSED - Ready for fast training!")
    print("="*70)
    return True

if __name__ == "__main__":
    import sys
    success = verify_setup()
    sys.exit(0 if success else 1)
