#
# gflownet-causet-genesis/causet_env.py
#
import torch
import networkx as nx
import numpy as np
from typing import Tuple
from gfn.env import Env
from gfn.states import States

class CausalSetEnv(Env):
    # ... (__init__ remains the same) ...
    def __init__(self, max_nodes: int, proxy, device='cpu'):
        self.max_nodes = max_nodes
        self.proxy = proxy
        self._device = torch.device(device)

        self.max_edges = (max_nodes * (max_nodes - 1)) // 2
        self.state_dim = 1 + self.max_edges

        _dummy_action_int = self.max_edges + 1
        _exit_action_int = self.max_edges + 1

        s0_tensor = torch.zeros((self.state_dim,), dtype=torch.float, device=self._device)
        sf_tensor = torch.full((self.state_dim,), -1.0, dtype=torch.float, device=self._device)

        dummy_action_tensor = torch.tensor([_dummy_action_int], dtype=torch.long, device=self._device)
        exit_action_tensor = torch.tensor([_exit_action_int], dtype=torch.long, device=self._device)

        super().__init__(
            s0=s0_tensor,
            sf=sf_tensor,
            state_shape=(self.state_dim,),
            action_shape=(1,),
            dummy_action=dummy_action_tensor,
            exit_action=exit_action_tensor,
        )

        self.preprocessor = None

    def make_States_class(self) -> type[States]:
        """Required by torchgfn - return the States class to use"""
        return States



    def get_forward_masks(self, states: States) -> torch.Tensor:
        """Computes a boolean mask of valid forward actions for a batch of states."""
        # This implementation re-uses the is_action_valid logic but adapted for masks.
        # It's a placeholder; a more efficient, vectorized implementation would be better.

        batch_size = len(states)
        masks = torch.zeros((batch_size, self.action_dim), dtype=torch.bool, device=self._device)

        for i in range(batch_size):
            state = states.tensor[i]
            n = int(state.item())

            # If terminal state, no actions are valid (already handled by zeros init)
            if n >= self.max_nodes:
                continue

            # Action space is 0 to (n_max*2). Need to iterate through potential actions
            for action in range(self.action_dim):
                 # Check if the action is valid using existing logic
                 is_valid = self.is_action_valid(states[i].unsqueeze(0), torch.tensor([action], device=self._device))
                 if is_valid.item():
                     masks[i, action] = True

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

    def _calculate_forward_masks(self, states: States) -> torch.Tensor:
        """Computes a boolean mask of valid forward actions for a batch of states."""
        batch_size = len(states)
        # Initialize with zeros (invalid)
        masks = torch.zeros((batch_size, self.action_dim), dtype=torch.bool, device=self._device)

        # Vectorized calculation is complex, use loop for clarity and correctness
        for i in range(batch_size):
            state = states.tensor[i]
            n = int(state.item())

            if n >= self.max_nodes:
                continue

            # Valid actions: connection decisions for existing nodes (0 to n-1) and EOS (-1 which maps to index self.action_dim-1 if we map correctly)
            # In your current setup, actions are complex (node_idx * 2 + decision) + EOS action
            # The action map needs to be consistent.
            # Assuming action space mapping is correct (0 to self.action_dim - 1)

            # Mark all actions that correspond to valid node indices as possible
            # We assume for N=4, action_dim=10 (9 edges + 1 EOS)
            # This logic needs to align with how you defined the action space previously (e.g. action < 0 was EOS in step())

            # Simple fix: The is_action_valid should be used here if it's correct

            # For debugging, we use a broad mask:
            # For states where n < max_nodes, any action that doesn't immediately fail should be masked True
            # This might cause issues if invalid actions are sampled, but should pass the 'has attribute' check

            # Let's assume the previous is_action_valid logic works correctly for single states:
            for action_idx in range(self.action_dim):
                # Check validity for single state
                is_valid = self.is_action_valid(states[i].unsqueeze(0), torch.tensor([action_idx], device=self._device))
                if is_valid.item():
                    masks[i, action_idx] = True

        return masks

    def _decode_state(self, state_tensor: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """Extract current n and edge adjacency from state tensor"""
        if state_tensor.dim() > 1:
             state_tensor = state_tensor.squeeze(0)

        # Ensure we always access index 0 safely
        if state_tensor.dim() == 0:
             n = int(state_tensor.item())
             edges = torch.tensor([])
        else:
             n = int(state_tensor[0].item()) # Use index 0
             edges = state_tensor[1:]
        return n, edges



    def step(self, states: States, actions: torch.Tensor) -> States:
        """
        Apply actions to states.
        """
        new_states = states.clone()

        for i in range(len(states)):
            # ... (loop logic remains the same) ...
            if states.is_sink_state[i]: continue
            state = states.tensor[i]; action = actions[i].item(); n = int(state.item())

            if n >= self.max_nodes: new_states.tensor[i] = self.sf.tensor; continue

            if action < 0: new_states.tensor[i] = n + 1
            else:
                node_idx = action // 2; decision = action % 2
                if node_idx < n and decision == 1:
                    edge_idx = self._get_edge_index(node_idx, n) + 1
                    new_states.tensor[i][edge_idx] = 1.0

        # FIX: Explicitly attach the forward_masks attribute to the resulting States object
        new_states.forward_masks = self._calculate_forward_masks(new_states)

        return new_states

    def is_action_valid(self, states: States, actions: torch.Tensor,
                       backward: bool = False) -> torch.Tensor:
        # ... (is_action_valid implementation remains the same, used by _calculate_forward_masks) ...
        """Check if actions are valid for given states"""
        if backward:
            return torch.zeros(len(states), dtype=torch.bool, device=self._device)

        valid = torch.ones(len(states), dtype=torch.bool, device=self._device)

        for i in range(len(states)):
            state = states.tensor[i]
            n = int(state.item())
            action = actions[i].item()

            if n >= self.max_nodes:
                valid[i] = False
                continue

            if action >= 0:
                node_idx = action // 2
                if node_idx >= n:
                    valid[i] = False

        return valid

    def backward_step(self, states: States, actions: torch.Tensor) -> States:
        """Backward sampling not implemented for causal sets"""
        raise NotImplementedError("Backward sampling not supported")


    def get_log_reward(self, final_states: States) -> torch.Tensor:
        """
        Compute log reward (negative energy) for final states.
        """
        rewards = []

        for i in range(len(final_states)):
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
