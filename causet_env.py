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
# --- FIX for 'Actions' AttributeError & ImportError ---
# Import the base 'Actions' class from gfn.actions
from gfn.actions import Actions
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
        # --- DEBUG ---
        # print(f"[DEBUG] CausalSetEnv.__init__: Initializing environment with max_nodes={max_nodes}")
        # -----------
        super().__init__()
        self.max_nodes = max_nodes
        self.proxy = proxy

        # --- FIX for 'AttributeError: can't set attribute 'device'' ---
        # The base 'Env' class defines 'device' as a property.
        # We must set the internal attribute 'self._device' to avoid collision.
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

        # --- FIX for 'Actions' AttributeError ---
        # We must manually set the attributes that gfn.Env.__init__
        # would normally set for the sampler to work.
        # We use the base 'Actions' class.
        self.Actions = Actions

        # n_actions = len(self.action_space). The sampler/policy
        # deals with [0, 1]. The eos_action (-1) is handled by
        # our custom agent and env.step() logic.
        self.n_actions = len(self.action_space)
        self.policy_output_dim = self.n_actions

        # --- FIX for 'action_shape' AttributeError ---
        # We must MANUALLY set the required class attributes
        # on the 'Actions' class we've assigned to this env.
        # The docs state action_shape is (1,) for discrete envs.
        self.Actions.action_shape = (1,)

        # --- FIX for 'torch.Size([])' error ---
        # The sampler requires dummy and exit actions to have a shape
        # matching 'action_shape'. We wrap in a list to make shape [1].
        self.Actions.dummy_action = torch.tensor([self.eos_action], device=self._device)
        self.Actions.exit_action = torch.tensor([self.eos_action], device=self._device)
        # ---

        # --- FIX for 'check_action_validity' AttributeError ---
        # The sampler checks this flag. Since our agent handles masking,
        # we can disable the env's check.
        self.check_action_validity = False

        # ---
        # --- FIX for Vectorized Env Flag ---
        # We must tell the library we are *not* vectorized,
        # so it correctly calls the non-vectorized `_step`
        # which iterates and calls our `step` -> `_step_single`.
        self.is_vectorized = False
        # ---
        # ---

        # 2. Define the source state as a raw tuple.
        source_tuple = (0, (), ())
        self.source = source_tuple

        # --- THE CORRECT FIX for the RuntimeError ---
        # Set the class-level template for ObjectStates so that
        # make_initial_states (called by the sampler) knows
        # what the source state looks like.
        self.States.set_source_state_template(source_tuple)
        # ------------------------------------------

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
        # --- DEBUG ---
        # print(f"[DEBUG] CausalSetEnv.__init__: Setting self.States.state_shape = {self.state_shape}")
        # -----------
        self.States.state_shape = self.state_shape

        # 3. Define s0 (source state object)
        #    This call now uses the 'ObjectStates' constructor,
        #    which will skip the broken assertion.
        # --- DEBUG ---
        # print(f"[DEBUG] CausalSetEnv.__init__: Creating self.s0...")
        # -----------
        # --- DEVICE FIX ---
        # Pass the correct device to the States constructor.
        self.s0 = self.States([source_tuple], device=self._device)
        # ---
        # --- DEBUG ---
        # print(f"[DEBUG] CausalSetEnv.__init__: self.s0 created. shape={self.s0.shape}")
        # -----------


        # 3a. MANUALLY SET THE CLASS ATTRIBUTE for s0.
        #     Now that 'self.s0' exists, we set it as the class
        #     attribute so that 'sf' (and other states) can
        #     be created and validated against it.
        # --- DEBUG ---
        # print(f"[DEBUG] CausalSetEnv.__init__: Setting self.States.s0 = self.s0")
        # -----------
        self.States.s0 = self.s0

        # 4. Define sf (sink state object)
        #    This call will use the 'ObjectStates' constructor,
        #    which will now safely find 'self.States.s0' and succeed.
        # --- DEBUG ---
        # print(f"[DEBUG] CausalSetEnv.__init__: Creating self.sf...")
        # -----------
        # --- DEVICE FIX ---
        # Pass the correct device to the States constructor.
        self.sf = self.States([self.eos_action], device=self._device)
        # ---
        # --- DEBUG ---
        # print(f"[DEBUG] CausalSetEnv.__init__: self.sf created. shape={self.sf.shape}")

        # --- FIX for 'is_sink_state' CRITICAL ERROR ---
        # We must set the CLASS attribute for sf, just like we did for s0.
        self.States.sf = self.sf
        # print(f"[DEBUG] CausalSetEnv.__init__: Setting self.States.sf = self.sf")
        # ---

        # print(f"[DEBUG] CausalSetEnv.__init__: Environment initialization complete.")
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

        # No transitivity constraints found, both actions are valid
        return self.mask_valid

    # ---
    # --- THIS IS THE FIX ---
    #
    # The traceback shows the sampler calls `env._step` (non-vectorized),
    # which iterates and calls `env.step` (single-state).
    # Your `step` was vectorized, causing the error.
    #
    # We rename your logic to `_step_single` and call it from `step`,
    # which is defined in the `FactorEnv` base class.
    # This correctly implements the non-vectorized API the
    # sampler is using.

    def _step_single(self, state: Any, action: Any) -> tuple[Any, Any, bool]:
        """
        Applies a single action to a single state.
        This is the original logic from your `step` function,
        but corrected to be single-state.
        """

        # ---
        # --- BUG FIX (from user log) ---
        # The 'action' passed by the sampler's non-vectorized loop
        # is an `Actions` object (e.g., of shape [1, 1]).
        # We must extract the Python integer value.
        try:
            action_int = action.tensor.item()
        except AttributeError:
            # Fallback if it's already an int (e.g., from test_causet.py)
            action_int = int(action)
        except Exception:
            # Broader fallback
            action_int = int(action)
        # ---
        # ---

        # `state` is a Python tuple
        n, edges, partial_v = state
        factor_idx = len(partial_v)
        num_factors = self.get_num_factors(state)
        is_done = False
        action_taken = action_int # Use the int

        if factor_idx == num_factors:
            # Stage is complete. 'action_int' is the 0 forced by the agent.
            new_n = n + 1
            new_node = n

            new_edges_list = list(edges)
            for j, v_j in enumerate(partial_v): # Use the state's partial_v
                if v_j == 1:
                    new_edges_list.append((j, new_node))
            new_edges = tuple(new_edges_list)

            new_state = (new_n, new_edges, ())

            if new_n == self.max_nodes:
                is_done = True

            self._graph_cache = {}

            # The single-state step must return the `eos_action`
            # to signal a non-factor step.
            action_taken = self.eos_action

        else:
            # This is a FACTOR DECISION step.
            # 'action_int' is the factor decision (0 or 1).
            new_partial_v = partial_v + (action_int,) # <--- USE THE INT
            new_state = (n, edges, new_partial_v)
            # action_taken is already correct (0 or 1)

        return new_state, action_taken, is_done

    # ---
    # --- END OF FIX ---
    # ---

    def get_log_reward(self, final_states: States) -> torch.Tensor:
        """
        Compute the log reward (unexponentiated energy) for a batch.
        This is called by the GFlowNet algorithm.
        """
        rewards = []

        # --- FIX ---
        # Extract raw tuples from the States object, similar to FactorPreAggAgent
        # --- SYNTAX ERROR FIX: Added missing colon ---
        try:
        # ---
            # --- DEBUG ---
            # Use .states_list, which is the correct attribute
            raw_states: List[tuple] = final_states.states_list
            # ---
        except AttributeError:
            try:
                raw_states: List[tuple] = list(final_states.tensor.flat)
            except AttributeError:
                raw_states: List[tuple] = list(final_states)
        except Exception as e:
            # --- DEBUG ---
            print(f"[DEBUG] CausalSetEnv.get_log_reward: CRITICAL ERROR extracting states. States object: {final_states}")
            # -----------
            raise ValueError(
                f"CausalSetEnv could not extract raw states (tuples) "
                f"from the gfn.States object. Error: {e}"
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
        # Return a tensor on the correct device (use self._device)
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
