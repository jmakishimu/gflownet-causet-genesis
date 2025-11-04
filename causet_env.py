#
# gflownet-causet-genesis/causet_env.py
# CORRECTED VERSION - Fixes -inf loss issue
#
import torch
import networkx as nx
import numpy as np
from typing import Tuple
from gfn.env import Env
from gfn.states import States

class CausalSetEnv(Env):
    def __init__(self, max_nodes: int, proxy, device='cpu'):
        self.max_nodes = max_nodes
        self.proxy = proxy
        self._device = torch.device(device)

        self.max_edges = (max_nodes * (max_nodes - 1)) // 2
        self.state_dim = 1 + self.max_edges

        # Action space:
        # 0 ... (max_nodes * 2 - 1): Connection decisions
        # (max_nodes * 2): "Add-node" action
        # (max_nodes * 2 + 1): GFlowNet "exit_action"
        self.n_actions = max_nodes * 2 + 2

        s0_tensor = torch.zeros((self.state_dim,), dtype=torch.float, device=self._device)
        sf_tensor = torch.full((self.state_dim,), -1.0, dtype=torch.float, device=self._device)

        dummy_action_tensor = torch.tensor([self.n_actions], dtype=torch.long, device=self._device)
        exit_action_tensor = torch.tensor([self.n_actions - 1], dtype=torch.long, device=self._device)

        super().__init__(
            s0=s0_tensor,
            sf=sf_tensor,
            state_shape=(self.state_dim,),
            action_shape=(1,),
            dummy_action=dummy_action_tensor,
            exit_action=exit_action_tensor,
        )

        self.preprocessor = None
        self.States = self.make_States_class()

    def make_States_class(self) -> type[States]:
        """Required by torchgfn - return the States class to use"""
        env = self

        class CausalSetStates(States):
            """Custom States class for CausalSet environment with forward_masks support"""

            state_shape = env.state_shape
            s0 = env.s0
            sf = env.sf

            def __init__(self, tensor: torch.Tensor):
                if tensor.dim() == 1:
                    tensor = tensor.unsqueeze(0)

                super().__init__(tensor)

                self.forward_masks = None
                self.backward_masks = None

                env.update_masks(self)

            def make_random_states_tensor(self, batch_shape: tuple) -> torch.Tensor:
                """Generate random state tensors"""
                return env.s0.repeat(*batch_shape, 1)

        return CausalSetStates

    def reset(self, batch_shape: tuple = (1,), random: bool = False):
        """Resets the environment and returns initial states."""
        if random:
            states_tensor = self.make_random_states_tensor(batch_shape)
        else:
            states_tensor = self.s0.repeat(*batch_shape, 1)

        states = self.States(states_tensor)
        return states

    def make_random_states_tensor(self, batch_shape: tuple[int, ...]) -> torch.Tensor:
        """Generate random states tensor for the environment"""
        return self.s0.repeat(*batch_shape, 1)

    def update_masks(self, states: States) -> None:
        """
        Updates forward_masks and backward_masks on the states object.
        This is required by torchgfn for discrete environments.
        """
        states.forward_masks = self._calculate_forward_masks(states)
        states.backward_masks = self._calculate_backward_masks(states)

    def _calculate_forward_masks(self, states: States) -> torch.Tensor:
        """
        Computes a boolean mask of valid forward actions for a batch of states.

        CRITICAL: Must handle sink state (n < 0) to prevent -inf in forward policy.
        """
        if states.batch_shape:
            batch_size = states.batch_shape[0]
        else:
            batch_size = 1

        masks = torch.zeros((batch_size, self.n_actions), dtype=torch.bool, device=self._device)

        for i in range(batch_size):
            if states.tensor.dim() == 1:
                state = states.tensor
            else:
                state = states.tensor[i]

            n = int(state[0].item())

            # CRITICAL FIX #1: Handle sink state for forward policy
            if n < 0:
                # At s_f, only the exit action should be valid
                # This prevents -inf when pf evaluates s_f
                masks[i, self.n_actions - 1] = True
                continue

            # If we've reached max nodes, ONLY the GFN exit action is valid
            if n >= self.max_nodes:
                masks[i, self.n_actions - 1] = True
                continue

            # When n=0, we can only add the first node
            if n == 0:
                masks[i, self.n_actions - 2] = True  # "Add-node" action
                continue

            # For n > 0, we have connection decisions for existing nodes
            for node_idx in range(n):
                masks[i, node_idx * 2] = True      # Decision: don't connect
                masks[i, node_idx * 2 + 1] = True  # Decision: connect

            # The "add-node" action is always valid (until n = max_nodes)
            masks[i, self.n_actions - 2] = True

        return masks

    def _calculate_backward_masks(self, states: States) -> torch.Tensor:
        """
        Computes a boolean mask of valid backward actions for a batch of states.

        CRITICAL FIX: For trajectory balance to work, backward masks must be:
        1. All TRUE at sink state s_f (pb can assign uniform distribution)
        2. Valid parent actions at all other states

        The key insight: torchgfn's TB loss evaluates pb at EVERY state in the
        trajectory, including s_f. Since s_f can be reached from any terminal
        state via exit_action, we need to allow pb to assign valid probabilities.

        Solution: Set ALL backward actions to TRUE at s_f. This allows pb to
        compute a valid (uniform) distribution, preventing -inf log probs.
        """
        if states.batch_shape:
            batch_size = states.batch_shape[0]
        else:
            batch_size = 1

        # n_actions - 1 because backward_masks don't include the GFN exit action
        masks = torch.zeros((batch_size, self.n_actions - 1), dtype=torch.bool, device=self._device)

        for i in range(batch_size):
            if states.tensor.dim() == 1:
                state = states.tensor
            else:
                state = states.tensor[i]

            n = int(state[0].item())

            # CRITICAL FIX: At s_f, allow ALL backward actions
            if n < 0:
                # Set all actions to True - pb will compute uniform distribution
                # This prevents -inf log probs when TB loss evaluates pb(s_f, action)
                masks[i, :] = True
                continue

            # s0 (n=0) has no parents in our construction
            if n == 0:
                continue

            # If n > 0:
            # Parent 1: state n-1 (via action "add-node")
            masks[i, self.n_actions - 2] = True  # "add-node" action

            # Parent 2: state n with different edges (via connection actions)
            for node_idx in range(n):
                action_no_connect = node_idx * 2
                action_connect = node_idx * 2 + 1

                if action_no_connect < (self.n_actions - 1):
                    masks[i, action_no_connect] = True

                if action_connect < (self.n_actions - 1):
                    masks[i, action_connect] = True

        return masks

    def _get_edge_index(self, i: int, j: int) -> int:
        """Convert node pair (i,j) where i<j to edge index in state vector"""
        if i >= j:
            raise ValueError(f"Invalid edge: i={i} must be < j={j}")
        return i * self.max_nodes - (i * (i + 1)) // 2 + (j - i - 1)

    def _state_to_graph(self, state_tensor: torch.Tensor) -> nx.DiGraph:
        """Convert state tensor to networkx DiGraph"""
        n, edges = self._decode_state(state_tensor)

        g = nx.DiGraph()
        g.add_nodes_from(range(n))

        if n > 0:
            for i in range(n):
                for j in range(i + 1, n):
                    edge_idx = self._get_edge_index(i, j)
                    if edge_idx < len(edges) and edges[edge_idx].item() > 0.5:
                        g.add_edge(i, j)

        tc = nx.transitive_closure(g)
        return tc

    def _decode_state(self, state_tensor: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """Extract current n and edge adjacency from state tensor"""
        if state_tensor.dim() > 1:
            state_tensor = state_tensor.squeeze(0)

        if state_tensor.dim() == 0:
            n = int(state_tensor.item())
            edges = torch.tensor([], device=self._device)
        else:
            n = int(state_tensor[0].item())
            edges = state_tensor[1:]
        return n, edges

    def step(self, states: States, actions: torch.Tensor) -> States:
        """Apply actions to states and return new states with updated masks."""
        if hasattr(actions, 'tensor'):
            actions_tensor = actions.tensor
        else:
            actions_tensor = actions

        new_tensor = states.tensor.clone()

        if states.batch_shape:
            batch_size = states.batch_shape[0]
        else:
            batch_size = 1

        for i in range(batch_size):
            if states.is_sink_state is not None and states.is_sink_state[i]:
                continue

            if new_tensor.dim() == 1:
                state = new_tensor
                action = actions_tensor.item() if actions_tensor.dim() == 0 else actions_tensor[0].item()
            else:
                state = new_tensor[i]
                action = actions_tensor[i].item()

            n = int(state[0].item())

            if n >= self.max_nodes:
                if action == self.n_actions - 1:
                    if new_tensor.dim() == 1:
                        new_tensor = self.sf.clone()
                    else:
                        new_tensor[i] = self.sf.clone()
                continue

            # "Add-node" action
            if action == self.n_actions - 2:
                state[0] = n + 1

            # GFN-exit action
            elif action == self.n_actions - 1:
                if new_tensor.dim() == 1:
                    new_tensor = self.sf.clone()
                else:
                    new_tensor[i] = self.sf.clone()

            # Connection decision actions
            elif action < self.n_actions - 2 and n > 0:
                node_idx = action // 2
                decision = action % 2
                if node_idx < n and decision == 1:
                    edge_idx = self._get_edge_index(node_idx, n) + 1
                    if edge_idx < len(state):
                        state[edge_idx] = 1.0

        new_states = self.States(new_tensor)
        return new_states

    def is_action_valid(self, states: States, actions: torch.Tensor,
                       backward: bool = False) -> bool:
        """
        Check if actions are valid for given states.

        IMPORTANT: Returns a single boolean (not a tensor) as required by torchgfn.
        """
        if hasattr(actions, 'tensor'):
            actions_tensor = actions.tensor
        else:
            actions_tensor = actions

        if backward:
            return False

        batch_size = states.batch_shape[0] if states.batch_shape else 1
        valid = torch.ones(batch_size, dtype=torch.bool, device=self._device)

        for i in range(batch_size):
            if states.tensor.dim() == 1:
                state = states.tensor
                action = actions_tensor.item() if actions_tensor.dim() == 0 else actions_tensor[0].item()
            else:
                state = states.tensor[i]
                action = actions_tensor[i].item()

            n = int(state[0].item())

            if action >= self.n_actions:
                valid[i] = False
                continue

            # Sink state: only exit action is valid
            if n < 0:
                valid[i] = (action == self.n_actions - 1)
                continue

            # Terminal state: only exit action is valid
            if n >= self.max_nodes:
                valid[i] = (action == self.n_actions - 1)
                continue

            # Initial state: only add-node is valid
            if n == 0:
                valid[i] = (action == self.n_actions - 2)
                continue

            # Intermediate states
            if action == self.n_actions - 1:
                # Exit action is invalid (must complete the causet first)
                valid[i] = False
            elif action == self.n_actions - 2:
                # Add-node is always valid
                valid[i] = True
            else:
                # Connection decisions
                node_idx = action // 2
                if node_idx >= n:
                    valid[i] = False
                else:
                    valid[i] = True

        # Return scalar boolean as required by torchgfn
        return valid.all().item()

    def backward_step(self, states: States, actions: torch.Tensor) -> States:
        """Backward sampling not implemented for causal sets"""
        raise NotImplementedError("Backward sampling not supported")

    def get_log_reward(self, final_states: States) -> torch.Tensor:
        """Compute log reward (negative energy) for final states."""
        batch_size = final_states.batch_shape[0] if final_states.batch_shape else 1
        rewards = []

        for i in range(batch_size):
            if final_states.tensor.dim() == 1:
                state = final_states.tensor
            else:
                state = final_states.tensor[i]

            n = int(state[0].item())

            if n != self.max_nodes:
                # Incomplete trajectory - very low reward
                rewards.append(-1e10)
            else:
                # Convert to graph and get energy
                g = self._state_to_graph(state)
                energy = self.proxy.get_energy(g)
                # Log reward = -energy (lower energy = higher reward)
                rewards.append(-energy)

        return torch.tensor(rewards, dtype=torch.float, device=self._device)

    def log_reward(self, final_states: States) -> torch.Tensor:
        """Alias for get_log_reward to match torchgfn API"""
        return self.get_log_reward(final_states)
