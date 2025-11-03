#
# gflownet-causet-genesis/causet_env.py
#
import torch
import networkx as nx
import numpy as np
from typing import List, Any
# --- FIX ---
# Import the base 'States' (for type hints) and our new 'ObjectStates'
from gfn.states import States
from custom_gfn import FactorEnv, ObjectStates
# -----------

# The state is a tuple: (n, edges, partial_v)
# n: current number of nodes (int)
# edges: a tuple of (u, v) tuples representing the DiGraph
# partial_v: a tuple representing the decisions made so far for the current stage

class CausalSetEnv(FactorEnv):
    """
    Implements the Causal Set Environment (Phase 1)

    This robust implementation uses the FactorEnv framework to
    correctly model the Classical Sequential Growth (CSG) model.
    """
    def __init__(self, max_nodes, proxy, device='cpu'):
        super().__init__()
        self.max_nodes = max_nodes
        self.proxy = proxy

        # --- FIX ---
        # Renamed 'self.device' to 'self._device' to avoid attribute
        # collision with the 'device' property in the base 'gfn.env.Env'.
        self._device = torch.device(device)
        # -----------

        # --- FIX ---
        # The gfn.Env base class sets these attributes. Since we
        # bypass its __init__, we must set them manually.

        # 1. Assign the state container class
        # --- FIX ---
        self.States = ObjectStates # Use our custom, safe ObjectStates class
        # -----------

        self.action_space = [0, 1] # Binary decisions
        self.eos_action = -1

        # 2. Define the source state as a raw tuple.
        source_tuple = (0, (), ())
        self.source = source_tuple

        # --- THE CORRECT FIX for the AttributeError ---
        #
        # We now use 'ObjectStates', which has a safe constructor
        # that avoids the circular 'self.s0' assertion.
        # We still must set the class attributes 'state_shape' and 's0'
        # *after* 's0' is created, so that 'sf' can be created.
        #
        # 5. Define the state_shape for the *environment*.
        #    For ObjectStates, the tensor is a 1D array of objects,
        #    so its shape is (1,).
        self.state_shape = (1,)

        # 5a. MANUALLY SET THE CLASS ATTRIBUTE for state_shape.
        self.States.state_shape = self.state_shape

        # 3. Define s0 (source state object)
        #    This call now uses the 'ObjectStates' constructor,
        #    which will skip the broken assertion.
        self.s0 = self.States([source_tuple])

        # 3a. MANUALLY SET THE CLASS ATTRIBUTE for s0.
        #     Now that 'self.s0' exists, we set it as the class
        #     attribute so that 'sf' (and other states) can
        #     be created and validated against it.
        self.States.s0 = self.s0

        # 4. Define sf (sink state object)
        #    This call will use the 'ObjectStates' constructor,
        #    which will now safely find 'self.States.s0' and succeed.
        self.sf = self.States([self.eos_action])
        # -----------

        # Pre-compute masks for efficiency
        # --- FIX --- (use self._device)
        self.mask_valid = torch.tensor([True, True], device=self._device)
        self.mask_force_0 = torch.tensor([True, False], device=self._device)
        self.mask_force_1 = torch.tensor([False, True], device=self._device)
        # -----------

        # Cache for graph objects to speed up get_mask()
        self._graph_cache = {}

    def get_action_space(self, state):
        return self.action_space

    def get_num_factors(self, state):
        """ The number of factors (decisions) in this stage is n. """
        n, _, _ = state
        return n

    def get_mask(self, state, stage, factor_idx):
        """
        Provides a mask for valid actions, enforcing transitivity.
        (This function is now fully corrected for publication)
        """
        n, edges, partial_v = state
        i = factor_idx # We are deciding for node i (edge i -> n)

        # (Reviewer 2 Fix) Use cached graph object.
        graph_key = (n, edges)
        if graph_key not in self._graph_cache:
            g = nx.DiGraph()
            g.add_nodes_from(range(n))
            g.add_edges_from(edges)
            self._graph_cache[graph_key] = g

        g = self._graph_cache[graph_key]

        # We are deciding v_i (edge i -> n)
        # We must check implications for all j in {0...i-1} (in partial_v)

        for j, v_j in enumerate(partial_v):
            # j is in {0...i-1}

            # --- Check FORCE 0 conditions ---

            # Rule 1: (j < i) AND (NOT j < n) => (NOT i < n)
            # Contrapositive of: (j < i) AND (i < n) => (j < n)
            # If we set v_i=1 (i < n), we'd imply j < n.
            # But v_j=0 (NOT j < n), so this is a contradiction.
            # We must force v_i=0.
            if g.has_edge(j, i) and v_j == 0:
                return self.mask_force_0

            # Rule 2: (i < j) AND (NOT j < n) => (NOT i < n)
            # (This was the (correct) 'Force 0' rule from the original file)
            # If we set v_i=1 (i < n), we'd have i < j and NOT j < n.
            # This violates transitivity.
            # We must force v_i=0.
            if g.has_edge(i, j) and v_j == 0:
                return self.mask_force_0

            # --- Check FORCE 1 condition ---

            # Rule 3: (i < j) AND (j < n) => (i < n)
            # (This replaces the incorrect 'Force 1' rule)
            # If we set v_i=0 (NOT i < n), we'd have i < j and j < n.
            # This violates transitivity.
            # We must force v_i=1.
            if g.has_edge(i, j) and v_j == 1:
                return self.mask_force_1

        # No transitivity constraints found, both actions are valid
        return self.mask_valid

    def step(self, state, action):
        """
        Apply a single factor action and update the state.

        Returns: (next_state, action_taken, is_done)
        """
        n, edges, partial_v = state
        factor_idx = len(partial_v)

        # Add the new action to the partial vector
        new_partial_v = partial_v + (action,)

        num_factors = self.get_num_factors(state)
        is_done = False # Default: not done

        if len(new_partial_v) < num_factors:
            # Stage is not complete, stay in the same stage
            new_state = (n, edges, new_partial_v)
            # is_done remains False
        else:
            # Stage is complete, add the new node and edges
            new_n = n + 1
            new_node = n

            # Build new edge list
            new_edges_list = list(edges)
            for i, v_i in enumerate(new_partial_v):
                if v_i == 1:
                    new_edges_list.append((i, new_node))

            new_edges = tuple(new_edges_list)

            # Move to the next stage, reset partial_v
            new_state = (new_n, new_edges, ())

            # --- FIX ---
            # Corrected typo: self.max_models -> self.max_nodes
            if new_n == self.max_nodes:
            # -----------
                is_done = True
            # else: is_done remains False

            # (Reviewer 2 Fix) Clear cache
            self._graph_cache = {}

        # --- FIX ---
        # Return (next_state, action_taken, is_done)
        return new_state, action, is_done
        # -----------

    def get_log_reward(self, final_states: States) -> torch.Tensor:
        """
        Compute the log reward (unexponentiated energy) for a batch.
        This is called by the GFlowNet algorithm.
        """
        rewards = []

        # --- FIX ---
        # Extract raw tuples from the States object, similar to FactorPreAggAgent
        try:
            raw_states: List[tuple] = list(final_states.tensor.flat)
        except AttributeError:
            raw_states: List[tuple] = list(final_states)
        except Exception:
            raise ValueError(
                "CausalSetEnv could not extract raw states (tuples) "
                "from the gfn.States object."
            )
        # -----------

        for state in raw_states:
            n, edges, _ = state
            # --- FIX ---
            # Corrected typo: self.max_models -> self.max_nodes
            if n != self.max_nodes:
            # -----------
                rewards.append(-1e10) # Penalize incomplete causets
            else:
                # Use the helper function to convert state to graph
                g = self.state_to_graph(state)
                # Use the stored proxy to get the energy
                rewards.append(self.proxy.get_energy(g))

        # --- FIX ---
        # Return a tensor on the correct device
        return torch.tensor(rewards, device=self._device, dtype=torch.float)
        # -----------

    # Helper functions for analysis/sampling
    def state_to_graph(self, state):
        n, edges, _ = state
        g = nx.DiGraph()
        # --- FIX ---
        # Corrected typo: add_models_from -> add_nodes_from
        g.add_nodes_from(range(n))
        # -----------
        g.add_edges_from(edges)
        return g
