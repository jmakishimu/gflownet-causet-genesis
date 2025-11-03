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
    # ... (docstring remains the same) ...
    def __init__(self, max_nodes: int, proxy, device='cpu'):
        self.max_nodes = max_nodes
        self.proxy = proxy
        self._device = torch.device(device)

        # Calculate state dimension
        self.max_edges = (max_nodes * (max_nodes - 1)) // 2
        self.state_dim = 1 + self.max_edges  # [n, edge_0, edge_1, ...]

        # Action space: an integer representing which node to connect to, or EOS
        self.action_dim = self.max_edges + 1

        # Define dummy and exit actions (as integers first)
        _dummy_action_int = self.action_dim
        _exit_action_int = self.action_dim

        # Initialize s0 and sf tensors (they should be 1D tensors representing a single state)
        s0_tensor = torch.zeros((self.state_dim,), dtype=torch.float, device=self._device)
        sf_tensor = torch.full((self.state_dim,), -1.0, dtype=torch.float, device=self._device)

        # Convert dummy/exit actions to tensors BEFORE passing to super().__init__
        dummy_action_tensor = torch.tensor([_dummy_action_int], dtype=torch.long, device=self._device)
        exit_action_tensor = torch.tensor([_exit_action_int], dtype=torch.long, device=self._device)

        # Call parent init with all required arguments as tensors
        super().__init__(
            s0=s0_tensor,
            sf=sf_tensor,
            state_shape=(self.state_dim,), # Pass the shape of a single state
            action_shape=(1,),             # Actions are single-element tensors
            dummy_action=dummy_action_tensor,
            exit_action=exit_action_tensor,
        )

        # Store preprocessor for state features
        self.preprocessor = None

    def make_States_class(self) -> type[States]:
        """Required by torchgfn - return the States class to use"""
        return States

    def _get_edge_index(self, i: int, j: int) -> int:
        """Convert node pair (i,j) where i<j to edge index in state vector"""
        if i >= j:
            raise ValueError(f"Invalid edge: i={i} must be < j={j}")
        # Map (i,j) to flat index in upper triangular matrix
        # For max_nodes, we have edges: (0,1), (0,2), ..., (0,n-1), (1,2), ..., (n-2,n-1)
        return i * self.max_nodes - (i * (i + 1)) // 2 + (j - i - 1)

    def _decode_state(self, state_tensor: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """Extract current n and edge adjacency from state tensor"""

        # Ensure input is treated as a 1D tensor (handle potential batch dimension if it exists)
        if state_tensor.dim() > 1:
             state_tensor = state_tensor.squeeze(0)

        n = int(state_tensor[0].item())
        edges = state_tensor[1:]
        return n, edges

    def _state_to_graph(self, state_tensor: torch.Tensor) -> nx.DiGraph:
        """Convert state tensor to networkx DiGraph"""
        n, edges = self._decode_state(state_tensor)

        g = nx.DiGraph()
        g.add_nodes_from(range(n))

        # Reconstruct edges from state
        for i in range(n):
            for j in range(i + 1, n):
                edge_idx = self._get_edge_index(i, j)
                if edges[edge_idx].item() > 0.5:  # Edge exists
                    g.add_edge(i, j)

        # Ensure transitivity
        tc = nx.transitive_closure(g)
        return tc

    def step(self, states: States, actions: torch.Tensor) -> States:
        """
        Apply actions to states. This follows the CSG model:
        - At stage n, we make n binary decisions (one per existing node)
        - After n decisions, we add node n and transition to stage n+1
        """
        new_states = states.clone()

        for i in range(len(states)):
            if states.is_sink_state[i]:
                continue

            state = states.tensor[i]
            action = actions[i].item()
            n = int(state[0].item())

            # Check if we're at terminal state
            if n >= self.max_nodes:
                new_states[i] = self.sf.tensor[0]
                continue

            # Sequential action application
            # We need to track which decision we're making (0 to n-1)
            # This is encoded in the state itself

            # Simple approach: each step adds one connection decision
            # Action = 0 or 1 (no edge / edge to node i)
            # We track decision index in a separate way

            # For simplicity: action encodes both the node index and decision
            # action = node_idx * 2 + decision (0 or 1)
            # EOS action = -1

            if action < 0:  # EOS - move to next stage
                # Add new node
                new_state = state.clone()
                new_state[0] = n + 1
                new_states[i] = new_state
            else:
                # Apply edge decision
                node_idx = action // 2
                decision = action % 2

                if node_idx < n and decision == 1:
                    # Add edge from node_idx to n
                    edge_idx = self._get_edge_index(node_idx, n) + 1  # +1 for n offset
                    new_state = state.clone()
                    new_state[edge_idx] = 1.0
                    new_states[i] = new_state
                else:
                    # No edge, just continue
                    new_states[i] = state

        return new_states

    def backward_step(self, states: States, actions: torch.Tensor) -> States:
        """Backward sampling not implemented for causal sets"""
        raise NotImplementedError("Backward sampling not supported")

    def is_action_valid(self, states: States, actions: torch.Tensor,
                       backward: bool = False) -> torch.Tensor:
        """Check if actions are valid for given states"""
        if backward:
            return torch.zeros(len(states), dtype=torch.bool, device=self._device)

        valid = torch.ones(len(states), dtype=torch.bool, device=self._device)

        for i in range(len(states)):
            state = states.tensor[i]
            n = int(state[0].item())
            action = actions[i].item()

            # Terminal state - no valid actions
            if n >= self.max_nodes:
                valid[i] = False
                continue

            # Check action bounds
            if action >= 0:
                node_idx = action // 2
                if node_idx >= n:
                    valid[i] = False

        return valid

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
