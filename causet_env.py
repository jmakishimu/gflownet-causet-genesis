#
# gflownet-causet-genesis/causet_env.py
#
import torch
import networkx as nx
import numpy as np
from typing import List, Any
from gfn.states import States
from custom_gfn import FactorEnv, ObjectStates
# --- FIX: Import CustomActions ---
from custom_gfn import CustomActions
# ---

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

        self._device = torch.device(device)

        self.States = ObjectStates

        self.action_space = [0, 1]
        self.eos_action = -1

        # --- FIX: Use CustomActions instead of base Actions ---
        self.Actions = CustomActions
        # ---

        self.n_actions = len(self.action_space)
        self.policy_output_dim = self.n_actions

        # --- FIX: Set action_shape on CustomActions ---
        self.Actions.action_shape = (1,)
        # ---

        self.Actions.dummy_action = torch.tensor([self.eos_action], device=self._device)
        self.Actions.exit_action = torch.tensor([self.eos_action], device=self._device)

        self.check_action_validity = False
        self.is_vectorized = False

        source_tuple = (0, (), ())
        self.source = source_tuple

        self.States.set_source_state_template(source_tuple)

        self.state_shape = (1,)
        self.States.state_shape = self.state_shape

        self.s0 = self.States([source_tuple], device=self._device)
        self.States.s0 = self.s0

        self.sf = self.States([self.eos_action], device=self._device)
        self.States.sf = self.sf

        self.mask_valid = torch.tensor([True, True], device=self._device)
        self.mask_force_0 = torch.tensor([True, False], device=self._device)
        self.mask_force_1 = torch.tensor([False, True], device=self._device)

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
        """
        n, edges, partial_v = state
        i = factor_idx

        graph_key = (n, edges)
        if graph_key not in self._graph_cache:
            g = nx.DiGraph()
            g.add_nodes_from(range(n))
            g.add_edges_from(edges)
            self._graph_cache[graph_key] = g

        g = self._graph_cache[graph_key]

        for j, v_j in enumerate(partial_v):
            if g.has_edge(j, i) and v_j == 0:
                return self.mask_force_0

        return self.mask_valid

    def _step_single(self, state: Any, action: Any) -> tuple[Any, Any, bool]:
        """
        Applies a single action to a single state.
        """
        try:
            action_int = action.tensor.item()
        except AttributeError:
            action_int = int(action)
        except Exception:
            action_int = int(action)

        n, edges, partial_v = state

        if n == self.max_nodes:
            return state, self.eos_action, True

        factor_idx = len(partial_v)
        num_factors = self.get_num_factors(state)
        is_done = False
        action_taken = action_int

        if factor_idx == num_factors:
            new_n = n + 1
            new_node = n

            new_edges_list = list(edges)
            for j, v_j in enumerate(partial_v):
                if v_j == 1:
                    new_edges_list.append((j, new_node))
            new_edges = tuple(new_edges_list)

            new_state = (new_n, new_edges, ())

            print(f"[DEBUG] STAGE TRANSITION: new_n={new_n}, self.max_nodes={self.max_nodes}")

            if new_n == self.max_nodes:
                is_done = True
                print(f"[DEBUG] STOP CONDITION MET. Setting is_done = True.")

            self._graph_cache = {}
            action_taken = self.eos_action

        else:
            new_partial_v = partial_v + (action_int,)
            new_state = (n, edges, new_partial_v)

        return new_state, action_taken, is_done

    def get_log_reward(self, final_states: States) -> torch.Tensor:
        """
        Compute the log reward (unexponentiated energy) for a batch.
        """
        rewards = []

        try:
            raw_states: List[tuple] = final_states.states_list
        except AttributeError:
            try:
                raw_states: List[tuple] = list(final_states.tensor.flat)
            except AttributeError:
                raw_states: List[tuple] = list(final_states)
        except Exception as e:
            print(f"[DEBUG] CausalSetEnv.get_log_reward: CRITICAL ERROR extracting states. States object: {final_states}")
            raise ValueError(
                f"CausalSetEnv could not extract raw states (tuples) "
                f"from the gfn.States object. Error: {e}"
            )

        for state in raw_states:
            n, edges, _ = state
            if n != self.max_nodes:
                rewards.append(-1e10)
            else:
                g = self.state_to_graph(state)
                rewards.append(self.proxy.get_energy(g))

        return torch.tensor(rewards, device=self._device, dtype=torch.float)

    def state_to_graph(self, state):
        n, edges, _ = state
        g = nx.DiGraph()
        g.add_nodes_from(range(n))
        g.add_edges_from(edges)
        return g
