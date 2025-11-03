#
# test_causet.py
#
# Updated unit tests for tensor-based causet environment
#
import unittest
import torch
import networkx as nx
import numpy as np
import random

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

    # In test_causet.py

    def test_env_initialization(self):
        # ...
        # self.assertEqual(self.env.s0.shape[0], 1)  # <-- Remove/Comment this line
        self.assertEqual(self.env.s0.ndim, 1) # s0 should be a 1D tensor
        self.assertEqual(self.env.s0.shape[0], self.env.state_dim) # Shape should match state_dim


        # Check s0 is all zeros (no nodes, no edges)
        self.assertTrue(torch.allclose(self.env.s0, torch.zeros_like(self.env.s0)))

        # Check sf is marked as terminal
        self.assertEqual(self.env.sf.shape, self.env.s0.shape)

        print("✓ Environment initialized correctly")

    def test_state_to_graph_empty(self):
        print("\nTesting Empty Graph Conversion...")

        # s0 should give empty graph
        state = self.env.s0[0]  # Shape: [state_dim]
        g = self.env._state_to_graph(state)

        self.assertEqual(g.number_of_nodes(), 0)
        self.assertEqual(g.number_of_edges(), 0)

        print("✓ Empty graph conversion works")

    def test_state_to_graph_simple(self):
        print("\nTesting Simple Graph Conversion...")

        # Create state with 2 nodes and 1 edge (0->1)
        state = torch.zeros(self.env.state_dim)
        state[0] = 2  # n = 2 nodes

        # Add edge 0->1
        edge_idx = self.env._get_edge_index(0, 1)
        state[edge_idx + 1] = 1.0  # +1 because first element is n

        g = self.env._state_to_graph(state)

        self.assertEqual(g.number_of_nodes(), 2)
        self.assertTrue(g.has_edge(0, 1))

        print("✓ Simple graph conversion works")

    def test_edge_index_mapping(self):
        print("\nTesting Edge Index Mapping...")

        # For max_nodes=4, edges are:
        # (0,1)=0, (0,2)=1, (0,3)=2, (1,2)=3, (1,3)=4, (2,3)=5

        self.assertEqual(self.env._get_edge_index(0, 1), 0)
        self.assertEqual(self.env._get_edge_index(0, 2), 1)
        self.assertEqual(self.env._get_edge_index(1, 2), 3)
        self.assertEqual(self.env._get_edge_index(2, 3), 5)

        print("✓ Edge indexing works correctly")

    def test_log_reward_incomplete(self):
        print("\nTesting Reward for Incomplete State...")

        from gfn.states import States

        # Create state with only 2 nodes (incomplete)
        state_tensor = torch.zeros(1, self.env.state_dim)
        state_tensor[0, 0] = 2

        states = States(state_tensor, state_shape=self.env.state_shape)
        rewards = self.env.log_reward(states)

        # Incomplete states should have very low reward
        self.assertLess(rewards[0].item(), -1e9)

        print("✓ Incomplete states penalized correctly")

    def test_log_reward_complete(self):
        print("\nTesting Reward for Complete State...")

        from gfn.states import States

        # Create complete state with max_nodes
        state_tensor = torch.zeros(1, self.env.state_dim)
        state_tensor[0, 0] = self.env.max_nodes

        # Add some edges (doesn't matter for mock proxy)
        state_tensor[0, 1] = 1.0  # Edge 0->1

        states = States(state_tensor, state_shape=self.env.state_shape)
        rewards = self.env.log_reward(states)

        # Complete states should have finite reward
        self.assertTrue(torch.isfinite(rewards[0]))

        print("✓ Complete states get proper reward")


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
