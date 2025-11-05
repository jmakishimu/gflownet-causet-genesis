#
# gflownet-causet-genesis/causet_env.py
# STAGED CONSTRUCTION VERSION - No infinite loops possible
#
import torch
import networkx as nx
import numpy as np
from typing import Tuple
from gfn.env import Env
from gfn.states import States

print("✓ Loading STAGED causet_env.py - No infinite loops!")

class CausalSetEnv(Env):
    """
    Staged Construction Environment:

    Phase 1: Add all nodes sequentially (n=0 -> n=1 -> ... -> n=max_nodes)
    Phase 2: For each node i (from 1 to max_nodes-1), make edge decisions
             for all potential parent nodes (0 to i-1)

    This eliminates infinite loops by enforcing a strict construction order.

    State representation:
    [n, phase, edge_cursor, edge_0_1, edge_0_2, ..., edge_{n-2}_{n-1}]

    - n: current number of nodes (0 to max_nodes)
    - phase: 0 = adding nodes, 1 = making edge decisions
    - edge_cursor: which edge we're currently deciding (in phase 1)
    """
    def __init__(self, max_nodes: int, proxy, device='cpu'):
        self.max_nodes = max_nodes
        self.proxy = proxy
        self._device = torch.device(device)

        self.max_edges = (max_nodes * (max_nodes - 1)) // 2
        # State: [n, phase, edge_cursor, edges...]
        self.state_dim = 3 + self.max_edges

        # Action space:
        # 0: add_node (valid in phase 0)
        # 1: no_edge (valid in phase 1)
        # 2: yes_edge (valid in phase 1)
        # 3: exit_action (valid when all edges decided)
        self.n_actions = 4

        # Initial state: n=0, phase=0, edge_cursor=0
        s0_tensor = torch.zeros((self.state_dim,), dtype=torch.float, device=self._device)

        # Terminal state: n=-1 (marker for terminal)
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

        # Verify batched rewards are available
        if hasattr(proxy, 'get_bd_energy_batched'):
            print(f"✓ Environment using BATCHED GPU rewards")
        else:
            print(f"⚠ WARNING: Proxy missing batched rewards - will be slow!")

    def make_States_class(self) -> type[States]:
        """Required by torchgfn - return the States class to use"""
        env = self

        class CausalSetStates(States):
            """Custom States class for staged CausalSet environment"""

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
        """Updates forward_masks and backward_masks on the states object."""
        states.forward_masks = self._calculate_forward_masks(states)
        states.backward_masks = self._calculate_backward_masks(states)

    def _calculate_forward_masks(self, states: States) -> torch.Tensor:
        """
        Computes forward action masks for staged construction.

        Actions:
        0: add_node (valid in phase 0)
        1: no_edge (valid in phase 1 when deciding edges)
        2: yes_edge (valid in phase 1 when deciding edges)
        3: exit_action (valid when all edges decided)
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
            phase = int(state[1].item())
            edge_cursor = int(state[2].item())

            # Terminal state
            if n < 0:
                masks[i, 3] = True  # Only exit
                continue

            if phase == 0:
                # Phase 0: Adding nodes
                if n < self.max_nodes:
                    masks[i, 0] = True  # add_node action
                else:
                    # Shouldn't reach here, but if we do, allow exit
                    masks[i, 3] = True

            elif phase == 1:
                # Phase 1: Making edge decisions
                total_edges_to_decide = (n * (n - 1)) // 2

                if edge_cursor < total_edges_to_decide:
                    # Still have edges to decide
                    masks[i, 1] = True  # no_edge
                    masks[i, 2] = True  # yes_edge
                else:
                    # All edges decided, can exit
                    masks[i, 3] = True

        return masks

    def _calculate_backward_masks(self, states: States) -> torch.Tensor:
        """
        Backward masks for staged construction.
        This is a simplified version - proper backward would require
        tracking the exact reverse of forward actions.
        """
        if states.batch_shape:
            batch_size = states.batch_shape[0]
        else:
            batch_size = 1

        # For now, we disable backward sampling
        # Backward actions exclude the exit action, so n_actions - 1
        masks = torch.zeros((batch_size, self.n_actions - 1), dtype=torch.bool, device=self._device)

        # In principle, we could implement proper backward:
        # - In phase 1 with edge_cursor > 0: can undo edge decision
        # - In phase 1 with edge_cursor = 0: can go back to phase 0
        # - In phase 0 with n > 0: can remove node

        return masks

    def _get_edge_index(self, i: int, j: int) -> int:
        """Convert node pair (i,j) where i<j to edge index in state vector"""
        if i >= j:
            raise ValueError(f"Invalid edge: i={i} must be < j={j}")
        return i * self.max_nodes - (i * (i + 1)) // 2 + (j - i - 1)

    def _edge_cursor_to_pair(self, cursor: int, n: int) -> Tuple[int, int]:
        """
        Convert edge cursor to (i, j) pair.
        For a graph with n nodes, edges are ordered:
        (0,1), (0,2), ..., (0,n-1), (1,2), ..., (1,n-1), ..., (n-2,n-1)
        """
        edge_count = 0
        for i in range(n):
            for j in range(i + 1, n):
                if edge_count == cursor:
                    return (i, j)
                edge_count += 1
        raise ValueError(f"Invalid edge cursor {cursor} for n={n}")

    def _state_to_graph(self, state_tensor: torch.Tensor) -> nx.DiGraph:
        """Convert state tensor to networkx DiGraph"""
        n, phase, edge_cursor, edges = self._decode_state(state_tensor)

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

    def _decode_state(self, state_tensor: torch.Tensor) -> Tuple[int, int, int, torch.Tensor]:
        """Extract n, phase, edge_cursor, and edge adjacency from state tensor"""
        if state_tensor.dim() > 1:
            state_tensor = state_tensor.squeeze(0)

        if state_tensor.dim() == 0:
            n = int(state_tensor.item())
            phase = 0
            edge_cursor = 0
            edges = torch.tensor([], device=self._device)
        else:
            n = int(state_tensor[0].item())
            phase = int(state_tensor[1].item())
            edge_cursor = int(state_tensor[2].item())
            edges = state_tensor[3:]

        return n, phase, edge_cursor, edges

    def step(self, states: States, actions: torch.Tensor) -> States:
        """
        Apply staged construction actions to states.

        Phase 0: action 0 (add_node) -> increment n
                 When n reaches max_nodes, transition to phase 1

        Phase 1: action 1 (no_edge) -> set current edge to 0, advance cursor
                 action 2 (yes_edge) -> set current edge to 1, advance cursor
                 When cursor reaches total edges, can take exit action
        """
        if hasattr(actions, 'tensor'):
            actions_tensor = actions.tensor
        else:
            actions_tensor = actions

        # Clone the tensor to avoid in-place modifications
        new_tensor = states.tensor.clone()

        if states.batch_shape:
            batch_size = states.batch_shape[0]
        else:
            batch_size = 1

        for i in range(batch_size):
            if states.is_sink_state is not None and states.is_sink_state[i]:
                continue

            # Get action for this batch element
            if new_tensor.dim() == 1:
                action = actions_tensor.item() if actions_tensor.dim() == 0 else actions_tensor[0].item()
            else:
                action = actions_tensor[i].item()

            # Read current state values
            if new_tensor.dim() == 1:
                n = int(new_tensor[0].item())
                phase = int(new_tensor[1].item())
                edge_cursor = int(new_tensor[2].item())
            else:
                n = int(new_tensor[i, 0].item())
                phase = int(new_tensor[i, 1].item())
                edge_cursor = int(new_tensor[i, 2].item())

            # Exit action (action 3)
            if action == 3:
                # Only valid when all edges are decided
                total_edges = (n * (n - 1)) // 2
                if phase == 1 and edge_cursor >= total_edges:
                    if new_tensor.dim() == 1:
                        new_tensor = self.sf.clone()
                    else:
                        new_tensor[i] = self.sf.clone()
                continue

            # Phase 0: Adding nodes
            if phase == 0:
                if action == 0:  # add_node
                    new_n = n + 1

                    # Update state tensor directly
                    if new_tensor.dim() == 1:
                        new_tensor[0] = new_n
                        # Check if we've added all nodes
                        if new_n >= self.max_nodes:
                            new_tensor[1] = 1  # Transition to phase 1
                            new_tensor[2] = 0  # Reset edge cursor
                    else:
                        new_tensor[i, 0] = new_n
                        # Check if we've added all nodes
                        if new_n >= self.max_nodes:
                            new_tensor[i, 1] = 1  # Transition to phase 1
                            new_tensor[i, 2] = 0  # Reset edge cursor

            # Phase 1: Making edge decisions
            elif phase == 1:
                total_edges = (n * (n - 1)) // 2

                if edge_cursor < total_edges:
                    # Get the edge we're deciding
                    i_node, j_node = self._edge_cursor_to_pair(edge_cursor, n)
                    edge_idx = self._get_edge_index(i_node, j_node)

                    # Update edge value
                    if action == 1:  # no_edge
                        if new_tensor.dim() == 1:
                            new_tensor[3 + edge_idx] = 0.0
                        else:
                            new_tensor[i, 3 + edge_idx] = 0.0
                    elif action == 2:  # yes_edge
                        if new_tensor.dim() == 1:
                            new_tensor[3 + edge_idx] = 1.0
                        else:
                            new_tensor[i, 3 + edge_idx] = 1.0

                    # Advance cursor
                    if new_tensor.dim() == 1:
                        new_tensor[2] = edge_cursor + 1
                    else:
                        new_tensor[i, 2] = edge_cursor + 1

        new_states = self.States(new_tensor)
        return new_states

    def is_action_valid(self, states: States, actions: torch.Tensor,
                       backward: bool = False) -> bool:
        """Check if actions are valid for given states."""
        if hasattr(actions, 'tensor'):
            actions_tensor = actions.tensor
        else:
            actions_tensor = actions

        if backward:
            return False  # Backward not fully implemented

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
            phase = int(state[1].item())
            edge_cursor = int(state[2].item())

            if action >= self.n_actions:
                valid[i] = False
                continue

            if n < 0:
                # Terminal state - only exit valid
                valid[i] = (action == 3)
                continue

            if phase == 0:
                # Phase 0: only add_node valid
                valid[i] = (action == 0)
            elif phase == 1:
                total_edges = (n * (n - 1)) // 2
                if edge_cursor < total_edges:
                    # Edge decision phase - no_edge or yes_edge valid
                    valid[i] = (action == 1 or action == 2)
                else:
                    # All edges decided - only exit valid
                    valid[i] = (action == 3)

        return valid.all().item()

    def backward_step(self, states: States, actions: torch.Tensor) -> States:
        """Backward sampling not fully implemented for staged construction"""
        raise NotImplementedError("Backward sampling not supported in staged mode")

    def get_log_reward(self, final_states: States) -> torch.Tensor:
        """
        OPTIMIZED: Compute log reward using batched GPU operations.
        """
        batch_size = final_states.batch_shape[0] if final_states.batch_shape else 1

        # Use batched BD computation if available
        if self.proxy.reward_type == 'bd' and hasattr(self.proxy, 'get_bd_energy_batched'):
            # Batched GPU computation - MUCH FASTER
            energies = self.proxy.get_bd_energy_batched(
                final_states.tensor,
                self.max_nodes
            )
            # Log reward = -energy (lower energy = higher reward)
            return -energies
        else:
            # Fallback to sequential (for MMD ablation or old proxy)
            rewards = []
            for i in range(batch_size):
                if final_states.tensor.dim() == 1:
                    state = final_states.tensor
                else:
                    state = final_states.tensor[i]

                n = int(state[0].item())

                if n != self.max_nodes:
                    rewards.append(-1e10)
                else:
                    g = self._state_to_graph(state)
                    energy = self.proxy.get_energy(g)
                    rewards.append(-energy)

            return torch.tensor(rewards, dtype=torch.float, device=self._device)

    def log_reward(self, final_states: States) -> torch.Tensor:
        """Alias for get_log_reward to match torchgfn API"""
        return self.get_log_reward(final_states)
