#
# gflownet-causet-genesis/causet_env.py
#
# --- THIS FILE IS CORRECTED ---
#
# Fixes applied:
# 1. Decoupled GFlowNet 'exit_action' from the environment's 'add-node' action.
# 2. 'is_action_valid' returns valid.all().
# 3. Implemented _calculate_backward_masks.
# 4. CRITICAL FIX: _calculate_forward_masks and is_action_valid
#    now handle the sink state (n < 0) to prevent -inf logprobs.
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
        # (max_nodes * 2) or (self.n_actions - 2): "Add-node" action
        # (max_nodes * 2 + 1) or (self.n_actions - 1): GFlowNet "exit_action"
        self.n_actions = max_nodes * 2 + 2

        s0_tensor = torch.zeros((self.state_dim,), dtype=torch.float, device=self._device)
        sf_tensor = torch.full((self.state_dim,), -1.0, dtype=torch.float, device=self._device)

        dummy_action_tensor = torch.tensor([self.n_actions], dtype=torch.long, device=self._device)
        # The true GFlowNet exit action is the LAST action
        exit_action_tensor = torch.tensor([self.n_actions - 1], dtype=torch.long, device=self._device)

        super().__init__(
            s0=s0_tensor,
            sf=sf_tensor,
            state_shape=(self.state_dim,),
            action_shape=(1,),
            dummy_action=dummy_action_tensor,
            exit_action=exit_action_tensor, # Pass the correct exit action
        )

        self.preprocessor = None

        # Store reference to custom States class
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
                # Ensure tensor has correct shape
                # States expects tensor of shape (batch_size, *state_shape)
                if tensor.dim() == 1:
                    # Single state - add batch dimension
                    tensor = tensor.unsqueeze(0)

                # States.__init__ only takes tensor and infers batch_shape from it
                super().__init__(tensor)

                # Initialize masks - will be set by update_masks
                self.forward_masks = None
                self.backward_masks = None

                # Immediately update masks after creation
                env.update_masks(self)

            def make_random_states_tensor(self, batch_shape: tuple) -> torch.Tensor:
                """Generate random state tensors"""
                return env.s0.repeat(*batch_shape, 1)

        return CausalSetStates

    def reset(self, batch_shape: tuple = (1,), random: bool = False):
        """
        Resets the environment and returns initial states.
        This method is called by torchgfn's Sampler.
        """
        if random:
            states_tensor = self.make_random_states_tensor(batch_shape)
        else:
            # Start from s0
            states_tensor = self.s0.repeat(*batch_shape, 1)

        # Use our custom States class (it will add batch dimension if needed)
        states = self.States(states_tensor)

        return states

    def make_random_states_tensor(self, batch_shape: tuple[int, ...]) -> torch.Tensor:
        """Generate random states tensor for the environment"""
        return self.s0.repeat(*batch_shape, 1)

    def update_masks(self, states: States) -> None:
        """
        CRITICAL: Updates forward_masks and backward_masks on the states object.
        This is required by torchgfn for discrete environments.
        """
        states.forward_masks = self._calculate_forward_masks(states)
        states.backward_masks = self._calculate_backward_masks(states)

    def _calculate_forward_masks(self, states: States) -> torch.Tensor:
        """Computes a boolean mask of valid forward actions for a batch of states."""
        # Get the batch size
        if states.batch_shape:
            batch_size = states.batch_shape[0]
        else:
            batch_size = 1

        # Use new action space size
        masks = torch.zeros((batch_size, self.n_actions), dtype=torch.bool, device=self._device)

        for i in range(batch_size):
            # Handle both batched and unbatched states
            if states.tensor.dim() == 1:
                state = states.tensor
            else:
                state = states.tensor[i]

            n = int(state[0].item())

            # --- THIS IS THE FIX ---
            # Handle sink state (n < 0)
            if n < 0:
                # This is the sink state s_f.
                # The TB loss will call pb.log_prob(s_f, exit_action).
                # To prevent -inf, we must set the mask for exit_action to True.
                masks[i, self.n_actions - 1] = True
                continue
            # --- END FIX ---

            # If we've reached max nodes, ONLY the GFN exit action is valid
            if n >= self.max_nodes:
                masks[i, self.n_actions - 1] = True  # GFN exit action
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
        """Computes a boolean mask of valid backward actions for a batch of states."""
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

            if n <= 0:
                # s0 (n=0) or s_f (n=-1) have no parents.
                continue

            # If n > 0:
            # Parent 1: state n-1 (via action "add-node")
            masks[i, self.n_actions - 2] = True # "add-node" action

            # Parent 2: state n with different edges (via connection actions)
            for node_idx in range(n): # Actions up to n
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
        """
        Apply actions to states and return new states with updated masks.
        """
        # Extract tensor from Actions object if needed
        if hasattr(actions, 'tensor'):
            actions_tensor = actions.tensor
        else:
            actions_tensor = actions

        # Clone states to create new states
        new_tensor = states.tensor.clone()

        # Determine batch size
        if states.batch_shape:
            batch_size = states.batch_shape[0]
        else:
            batch_size = 1

        for i in range(batch_size):
            # Handle sink states - if already sink, stay sink
            if states.is_sink_state is not None and states.is_sink_state[i]:
                continue

            # Get state and action
            if new_tensor.dim() == 1:
                state = new_tensor
                action = actions_tensor.item() if actions_tensor.dim() == 0 else actions_tensor[0].item()
            else:
                state = new_tensor[i]
                action = actions_tensor[i].item()

            n = int(state[0].item())

            # If at or past terminal state, only valid action is GFN-exit
            if n >= self.max_nodes:
                if action == self.n_actions - 1: # GFN exit action
                    if new_tensor.dim() == 1:
                        new_tensor = self.sf.clone()
                    else:
                        new_tensor[i] = self.sf.clone()
                # Any other action is invalid, state remains unchanged (but will be masked)
                continue

            # "Add-node" action: finalize and add new node (increment n)
            if action == self.n_actions - 2:
                state[0] = n + 1

            # GFN-exit action (premature): transition to sink
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

        # Create new States object with updated tensor using custom States class
        new_states = self.States(new_tensor)

        return new_states

    def is_action_valid(self, states: States, actions: torch.Tensor,
                       backward: bool = False) -> torch.Tensor:
        """
        Check if actions are valid for given states
        """
        # Extract tensor from Actions object if needed
        if hasattr(actions, 'tensor'):
            actions_tensor = actions.tensor
        else:
            actions_tensor = actions

        if backward:
            # Return a single boolean
            return torch.tensor(False, device=self._device)

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

            # Check if action is out of bounds
            if action >= self.n_actions:
                valid[i] = False
                continue

            # --- FIX for sink state ---
            if n < 0:
                # s_f state. Only valid action is the exit action
                valid[i] = (action == self.n_actions - 1)
                continue
            # --- END FIX ---

            # If at terminal state, only GFN-exit is valid
            if n >= self.max_nodes:
                valid[i] = (action == self.n_actions - 1)
                continue

            # If n=0, only "add-node" is valid
            if n == 0:
                valid[i] = (action == self.n_actions - 2)
                continue

            # If 0 < n < max_nodes:
            # GFN-exit action is invalid
            if action == self.n_actions - 1:
                valid[i] = False
            # "Add-node" action is valid
            elif action == self.n_actions - 2:
                valid[i] = True
            # Connection decisions
            else:
                node_idx = action // 2
                if node_idx >= n:
                    valid[i] = False
                else:
                    valid[i] = True # Both (connect/no-connect) are valid

        # Return .all() as a scalar boolean
        return valid.all()

    def backward_step(self, states: States, actions: torch.Tensor) -> States:
        """Backward sampling not implemented for causal sets"""
        raise NotImplementedError("Backward sampling not supported")

    def get_log_reward(self, final_states: States) -> torch.Tensor:
        """
        Compute log reward (negative energy) for final states.
        """
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
