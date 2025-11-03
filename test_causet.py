import unittest
import torch
import networkx as nx
import numpy as np
import random

# Do not import States directly from gfn.states here
from causet_env import CausalSetEnv
from causet_reward import CausalSetRewardProxy

class MockProxy:
    """Mock proxy for testing environment"""
    def get_energy(self, g):
        return 0.0

class TestCausalSetEnv(unittest.TestCase):

    def setUp(self):
        self.proxy = MockProxy()
        self.env = CausalSetEnv(max_nodes=4, proxy=self.proxy, device='cpu')
        # The env object itself now has access to the correct States class via self.env.States

    def test_env_initialization(self):
        """Test if the environment initializes correctly"""
        self.assertIsInstance(self.env, CausalSetEnv)
        print("✓ Environment initialized correctly")

    def test_state_to_graph_empty(self):
        """Testing Empty Graph Conversion..."""
        state = self.env.s0.clone()
        g = self.env._state_to_graph(state)
        self.assertEqual(g.number_of_nodes(), 0)
        self.assertEqual(g.number_of_edges(), 0)
        print("✓ Empty graph conversion works")

    def test_log_reward_incomplete(self):
        """Testing Reward for Incomplete State..."""
        # Create a state tensor starting from s0
        state_tensor = self.env.s0.clone()
        # FIX: Assign value 2.0 to the *first index* of the tensor, not overwrite the variable itself
        state_tensor[0] = 2.0

        # Use the env's States class implicitly linked to its configuration
        states = self.env.States(state_tensor.unsqueeze(0))

        log_rewards = self.env.log_reward(states)
        self.assertTrue(torch.allclose(log_rewards, torch.tensor([-1e10])))
        print("✓ Incomplete state reward works")

    def test_log_reward_complete(self):
        """Testing Reward for Complete State..."""
        # Create a state tensor starting from s0
        state_tensor = self.env.s0.clone()
        # FIX: Assign value max_nodes to the *first index* of the tensor
        state_tensor[0] = float(self.env.max_nodes)

        # Mock proxy returns 0.0 energy, so reward should be 0.0

        # Use the env's States class
        states = self.env.States(state_tensor.unsqueeze(0))

        log_rewards = self.env.log_reward(states)
        self.assertTrue(torch.allclose(log_rewards, torch.tensor([0.0])))
        print("✓ Complete state reward works")


class TestRewardProxy(unittest.TestCase):

    def setUp(self):
        random.seed(42)
        np.random.seed(42)

    def test_bd_action_diamond_graph(self):
        print("\nTesting BD Action (Diamond Graph)...")

        # 4-node diamond: 0<1<3, 0<2<3
        g = nx.DiGraph()
        g.add_nodes_from(range(4))
        g.add_edges_from([(0,1), (0,2), (1,3), (2,3)])

        proxy = CausalSetRewardProxy(reward_type='bd')
        energy = proxy.get_bd_energy(g)

        # Manual calculation gives S_BD = -16.0
        self.assertAlmostEqual(energy, -16.0, places=1)

        print(f"✓ BD Action = {energy:.2f} (expected -16.0)")

    def test_mmd_antichain(self):
        print("\nTesting MMD for Antichain...")

        # Pure antichain (no edges)
        g = nx.DiGraph()
        g.add_nodes_from(range(4))

        proxy = CausalSetRewardProxy(reward_type='mmd', target_dim=4.0)

        # Antichain should have ratio ~0, giving d~1
        avg_ratio = proxy.calculate_avg_mmd_ratio(g, num_samples=500)
        self.assertAlmostEqual(avg_ratio, 0.0, places=2)

        avg_mmd = proxy.solve_mmd_from_ratio(avg_ratio)
        self.assertAlmostEqual(avg_mmd, 1.01, places=1)

        # Energy should be high: (1.01 - 4)^2 ~ 9
        energy = proxy.get_mmd_energy(g)
        self.assertGreater(energy, 8.0)

        print(f"✓ MMD = {avg_mmd:.2f}, Energy = {energy:.2f}")

    def test_mmd_chain(self):
        print("\nTesting MMD for 1D Chain...")

        # 1D chain: 0<1<2<3
        g = nx.DiGraph()
        g.add_nodes_from(range(4))
        g.add_edges_from([(0,1), (1,2), (2,3)])

        proxy = CausalSetRewardProxy(reward_type='mmd', target_dim=4.0)

        avg_ratio = proxy.calculate_avg_mmd_ratio(g, num_samples=500)
        avg_mmd = proxy.solve_mmd_from_ratio(avg_ratio)

        # 1D chain should give d~2
        self.assertAlmostEqual(avg_mmd, 2.0, places=1)

        # Energy: (2-4)^2 = 4
        energy = proxy.get_mmd_energy(g)
        self.assertAlmostEqual(energy, 4.0, places=1)

        print(f"✓ MMD = {avg_mmd:.2f}, Energy = {energy:.2f}")


def run_tests():
    """Run all tests with nice output"""

    print("="*60)
    print("RUNNING CAUSAL SET GFLOWNET TESTS")
    print("="*60)

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestCausalSetEnv))
    suite.addTests(loader.loadTestsFromTestCase(TestRewardProxy))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "="*60)
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*60)

    return result.wasSuccessful()


if __name__ == "__main__":
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)
