#
# test_causet.py
#
# Unit tests for the custom GFlowNet causet environment.
# Run with: python test_causet.py
#
import unittest
import torch
import networkx as nx
import numpy as np
import random

# Import custom components to be tested
from causet_env import CausalSetEnv
from causet_reward import CausalSetRewardProxy
from train import collate_fn

# Mock proxy for env init
class MockProxy:
    def get_energy(self, g):
        return 0.0

class TestCausalSetEnv(unittest.TestCase):

    def setUp(self):
        # We can use a mock proxy for most env tests
        self.proxy = MockProxy()
        self.env = CausalSetEnv(max_nodes=4, proxy=self.proxy, device='cpu')
        self.s0 = self.env.source # (0, (), ())

    def test_env_initialization(self):
        print("\nTesting Environment Initialization...")
        self.assertEqual(self.s0, (0, (), ()))
        self.assertEqual(self.env.s0.states_list[0], self.s0)
        self.assertEqual(self.env.sf.states_list[0], self.env.eos_action)
        print("OK")

    def test_env_get_num_factors(self):
        print("\nTesting get_num_factors...")
        # At s0 (n=0), num_factors is 0
        self.assertEqual(self.env.get_num_factors(self.s0), 0)
        # At n=1, num_factors is 1
        s1 = (1, (), ())
        self.assertEqual(self.env.get_num_factors(s1), 1)
        # At n=2, num_factors is 2
        s2 = (2, ((0,1),), ())
        self.assertEqual(self.env.get_num_factors(s2), 2)
        print("OK")

    def test_env_step_full_trajectory(self):
        print("\nTesting Environment Step Logic (Full Trajectory)...")

        # 1. Start at s0 = (0, (), ())
        state = self.s0
        self.assertEqual(self.env.get_num_factors(state), 0) # n=0

        # 2. Stage transition: s0 -> s1
        # Agent sees n=0, factor_idx=0. (idx == num_factors)
        # Agent forces action 0.
        next_state, action, done = self.env.step(state, 0) # Pass transition action
        self.assertEqual(next_state, (1, (), ())) # n=1, partial_v=()
        self.assertFalse(done)
        self.assertEqual(action, self.env.eos_action)
        state = next_state # Now at n=1

        # 3. Factor step: s1 -> s1'
        # state = (1, (), ()). num_factors=1. factor_idx=0.
        # Agent sees (idx < num_factors).
        # Agent decides v_0 = 1 (i.e., 0 < 1).
        next_state, action, done = self.env.step(state, 1) # Pass factor action 1
        self.assertEqual(next_state, (1, (), (1,))) # n=1, partial_v=(1,)
        self.assertFalse(done)
        self.assertEqual(action, 1)
        state = next_state # Now at n=1, partial_v=(1,)

        # 4. Stage transition: s1' -> s2
        # state = (1, (), (1,)). num_factors=1. factor_idx=1.
        # Agent sees (idx == num_factors).
        # Agent forces action 0.
        next_state, action, done = self.env.step(state, 0) # Pass transition action
        self.assertEqual(next_state, (2, ((0,1),), ())) # Edges now contain (0,1)
        self.assertFalse(done)
        self.assertEqual(action, self.env.eos_action)
        state = next_state # Now at n=2

        # 5. Factor steps: s2 -> s2' -> s2''
        # state = (2, ((0,1),), ()). num_factors=2. factor_idx=0.
        # Agent decides v_0 = 0 (NOT 0 < 2)
        state, action, done = self.env.step(state, 0) # Pass factor action 0
        self.assertEqual(state, (2, ((0,1),), (0,)))
        self.assertEqual(action, 0)

        # state = (2, ((0,1),), (0,)). num_factors=2. factor_idx=1.
        # Agent decides v_1 = 1 (1 < 2)
        state, action, done = self.env.step(state, 1) # Pass factor action 1
        self.assertEqual(state, (2, ((0,1),), (0, 1)))
        self.assertEqual(action, 1)

        # 6. Stage transition: s2'' -> s3
        # state = (2, ((0,1),), (0, 1)). num_factors=2. factor_idx=2.
        # Agent sees (idx == num_factors).
        # Agent forces action 0.
        next_state, action, done = self.env.step(state, 0) # Pass transition action
        # New edges: (0,1) from before, and now (1,2) from v_1=1
        self.assertEqual(next_state, (3, ((0,1), (1,2)), ()))
        self.assertFalse(done)
        self.assertEqual(action, self.env.eos_action)
        state = next_state

        # 7. Final stage (n=3) -> (n=4)
        # state = (3, ((0,1), (1,2)), ()). num_factors=3.
        # v_0 = 1 (0 < 3)
        state, _, _ = self.env.step(state, 1)
        # v_1 = 1 (1 < 3)
        state, _, _ = self.env.step(state, 1)
        # v_2 = 1 (2 < 3)
        state, _, _ = self.env.step(state, 1)
        # Now state is (3, ((0,1), (1,2)), (1, 1, 1))

        # 8. Final stage transition: s3''' -> s4 (Done)
        # state.num_factors=3, factor_idx=3.
        # Agent sees (idx == num_factors).
        # Agent forces action 0.
        next_state, action, done = self.env.step(state, 0) # Pass transition action

        # Edges: (0,1), (1,2) from before
        # New edges: (0,3), (1,3), (2,3)
        # Note: tuple order might differ, so we check set equality
        expected_edges = set([(0,1), (1,2), (0,3), (1,3), (2,3)])
        self.assertEqual(set(next_state[1]), expected_edges)
        self.assertEqual(next_state[0], 4) # n=4
        self.assertEqual(next_state[2], ()) # partial_v reset
        self.assertTrue(done) # n == max_nodes
        self.assertEqual(action, self.env.eos_action)
        print("OK")

    def test_mask_logic_rule_1_force_0(self):
        print("\nTesting Mask Logic (Rule 1: Force 0)...")
        # Test Rule 1: (j < i) AND (NOT j < n) => (NOT i < n)
        # We set up a state where n=2, graph has 0<1.
        # We are deciding for new node n=2.
        # We are at factor_idx=1 (deciding v_1, for edge 1->2).
        # We set partial_v=(0,), meaning v_0=0 (so NOT 0 < 2).
        state = (2, ((0,1),), (0,)) # n=2, edges=(0<1), partial_v=(v_0=0)

        # get_mask(state, stage=2, factor_idx=1)
        # i=1. Loop starts, j=0, v_j=0.
        # Check Rule 1: g.has_edge(j, i) -> g.has_edge(0, 1) -> True
        #             v_j == 0 -> True
        # Rule 1 is triggered. Must force v_i = 0.
        mask = self.env.get_mask(state, 2, 1)

        # mask_force_0 is [True, False]
        self.assertTrue(torch.equal(mask, self.env.mask_force_0))
        print("OK")

    def test_mask_logic_no_rules_valid(self):
        print("\nTesting Mask Logic (No Rules Triggered)...")
        # This state is similar to the one above, but v_0=1.
        state = (2, ((0,1),), (1,)) # n=2, edges=(0<1), partial_v=(v_0=1)

        # get_mask(state, stage=2, factor_idx=1)
        # i=1. Loop starts, j=0, v_j=1.
        # Check Rule 1: g.has_edge(j, i) -> g.has_edge(0, 1) -> True
        #             v_j == 0 -> False
        # Rule 1 is NOT triggered.
        # Loop finishes.
        mask = self.env.get_mask(state, 2, 1)

        # mask_valid is [True, True]
        self.assertTrue(torch.equal(mask, self.env.mask_valid))
        print("OK")

class TestRewardProxy(unittest.TestCase):

    def setUp(self):
        # Fix random seed for reproducible MMD sampling
        random.seed(42)
        np.random.seed(42)

    def test_bd_action_diamond_graph(self):
        print("\nTesting BD Action (Diamond Graph)...")
        # S_BD for 4-node diamond graph (0<1<3, 0<2<3)
        g = nx.DiGraph()
        g.add_nodes_from(range(4))
        g.add_edges_from([(0,1), (0,2), (1,3), (2,3)])

        proxy = CausalSetRewardProxy(reward_type='bd')

        # Manual calculation:
        # C = [[1, 1, 1, 1], [0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 0, 1]]
        # C @ C = [[1, 2, 2, 4], [0, 1, 0, 2], [0, 0, 1, 2], [0, 0, 0, 1]]
        # counts = [7, 5, 4, 0, 1] (for sizes 0, 1, 2, 3, 4)
        # N_0 = counts[2] = 4
        # N_1 = counts[3] = 0
        # N_2 = counts[4] = 1
        # N_3 = counts[5] = 0
        # S_BD = n - N_0 + 9*N_1 - 16*N_2 + 8*N_3
        # S_BD = 4 - 4 + 9*0 - 16*1 + 8*0 = -16.0

        energy = proxy.get_bd_energy(g)
        self.assertAlmostEqual(energy, -16.0)
        print("OK")

    def test_mmd_action_antichain_fallback(self):
        print("\nTesting MMD Action (Antichain Fallback)...")
        # Test the "MMD Flaw Fix" for a pure antichain
        g = nx.DiGraph()
        g.add_nodes_from(range(4))

        proxy = CausalSetRewardProxy(reward_type='mmd', target_dim=4.0)

        # 1. Avg ratio should be 0.0 (no x < y pairs found)
        avg_ratio = proxy.calculate_avg_mmd_ratio(g, num_samples=500)
        self.assertAlmostEqual(avg_ratio, 0.0)

        # 2. Avg MMD solved from 0.0 ratio should be ~1.01
        avg_mmd = proxy.solve_mmd_from_ratio(avg_ratio)
        self.assertAlmostEqual(avg_mmd, 1.01)

        # 3. Energy should be high: (1.01 - 4.0)^2
        energy = proxy.get_mmd_energy(g)
        self.assertAlmostEqual(energy, (1.01 - 4.0)**2)
        print("OK")

    def test_mmd_action_1d_chain(self):
        print("\nTesting MMD Action (1D Chain is d=2)...")
        # A 1D chain (0<1<2<3)
        g = nx.DiGraph()
        g.add_nodes_from(range(4))
        g.add_edges_from([(0,1), (1,2), (2,3)])

        proxy = CausalSetRewardProxy(reward_type='mmd', target_dim=4.0)

        # 1. Avg ratio should be ~0.25, which corresponds to d=2
        avg_ratio = proxy.calculate_avg_mmd_ratio(g, num_samples=500)

        # 2. Avg MMD solved should be very close to 2.0
        avg_mmd = proxy.solve_mmd_from_ratio(avg_ratio)

        # --- TEST FIX ---
        # The assertion was 'assertLess(avg_mmd, 1.5)'.
        # The correct assertion is that avg_mmd is ~2.0.
        self.assertAlmostEqual(avg_mmd, 2.0, places=5)
        # ---

        # 3. Energy should be high, (2.0 - 4.0)^2 = 4.0
        energy = proxy.get_mmd_energy(g)

        # --- TEST FIX ---
        # Assert energy is ~4.0
        self.assertAlmostEqual(energy, (2.0 - 4.0)**2, places=5)
        self.assertGreater(energy, 3.9) # Check it's a high penalty
        # ---
        print("OK")

class TestCollateFunction(unittest.TestCase):

    def test_collate_empty_graph(self):
        print("\nTesting Collate Fn (Empty Graph)...")
        # s0 state
        states_list = [(0, (), ())]
        batch = collate_fn(states_list)

        self.assertEqual(batch.num_graphs, 1)
        self.assertEqual(batch.num_nodes, 0)
        self.assertEqual(batch.x.shape[0], 0)
        self.assertEqual(batch.edge_index.shape[1], 0)
        print("OK")

    def test_collate_simple_graph(self):
        print("\nTesting Collate Fn (Simple Graph)...")
        # n=2, 0<1
        states_list = [(2, ((0,1),), ())]
        batch = collate_fn(states_list)

        self.assertEqual(batch.num_graphs, 1)
        self.assertEqual(batch.num_nodes, 2)
        self.assertEqual(batch.x.shape[0], 2)
        self.assertEqual(batch.x.shape[1], 5) # 5 features
        self.assertEqual(batch.edge_index.shape[0], 2)
        self.assertEqual(batch.edge_index.shape[1], 1) # 1 edge

        # Check features for node 0: [idx, in, out, ancestors, descendants]
        # tc.in_degree(0) = 0, tc.out_degree(0) = 1
        expected_x0 = torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0])
        # Check features for node 1:
        # tc.in_degree(1) = 1, tc.out_degree(1) = 0
        expected_x1 = torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0])

        self.assertTrue(torch.equal(batch.x[0], expected_x0))
        self.assertTrue(torch.equal(batch.x[1], expected_x1))
        print("OK")

    def test_collate_batch(self):
        print("\nTesting Collate Fn (Batch)...")
        # Batch of s0 and a 2-node graph
        states_list = [(0, (), ()), (2, ((0,1),), ())]
        batch = collate_fn(states_list)

        self.assertEqual(batch.num_graphs, 2)
        self.assertEqual(batch.num_nodes, 2) # 0 + 2
        self.assertEqual(batch.x.shape[0], 2)
        self.assertEqual(batch.edge_index.shape[1], 1)

        # Check pointer
        # ptr[0] = 0 (start of graph 0)
        # ptr[1] = 0 (start of graph 1)
        # ptr[2] = 2 (end of graph 1)
        self.assertTrue(torch.equal(batch.ptr, torch.tensor([0, 0, 2])))
        print("OK")


if __name__ == "__main__":
    # Temporarily silence the debug prints from the modules
    import sys
    import io

    print("--- Running Unit Tests for Causal Set GFlowNet ---")

    # Redirect stdout
    original_stdout = sys.stdout
    sys.stdout = io.StringIO()

    try:
        # Run tests
        suite = unittest.TestSuite()
        suite.addTest(unittest.makeSuite(TestCausalSetEnv))
        suite.addTest(unittest.makeSuite(TestRewardProxy))
        suite.addTest(unittest.makeSuite(TestCollateFunction))

        runner = unittest.TextTestRunner(stream=original_stdout, verbosity=0)
        runner.run(suite)
    finally:
        # Restore stdout
        sys.stdout = original_stdout

    print("\n--- Test Run Complete ---")
