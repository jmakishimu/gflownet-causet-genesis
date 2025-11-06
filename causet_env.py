#
# gflownet-causet-genesis/causet_env.py
# FIXED VERSION - Corrected reward shaping
#
import torch
import networkx as nx
import numpy as np
from typing import Tuple
from gfn.env import Env
from gfn.states import States

print("✓ Loading FIXED causet_env.py with CORRECTED REWARD SHAPING")

class CausalSetEnv(Env):
    """
    Staged Construction Environment with FIXED reward shaping.

    Key fix: Reward is now exp(-beta * S_BD^2), creating proper basin at S_BD=0
    """
    def __init__(self, max_nodes: int, proxy, device='cpu'):
        self.max_nodes = max_nodes
        self.proxy = proxy
        self._device = torch.device(device)

        self.max_edges = (max_nodes * (max_nodes - 1)) // 2
        self.state_dim = 3 + self.max_edges
        self.n_actions = 4

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

        if hasattr(proxy, 'get_bd_energy_batched'):
            print(f"✓ Environment using BATCHED GPU rewards with FIXED SHAPING")
        else:
            print(f"⚠ WARNING: Proxy missing batched rewards - will be slow!")

    def make_States_class(self) -> type[States]:
        env = self

        class CausalSetStates(States):
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
                return env.s0.repeat(*batch_shape, 1)

        return CausalSetStates

    def reset(self, batch_shape: tuple = (1,), random: bool = False):
        if random:
            states_tensor = self.make_random_states_tensor(batch_shape)
        else:
            states_tensor = self.s0.repeat(*batch_shape, 1)
        states = self.States(states_tensor)
        return states

    def make_random_states_tensor(self, batch_shape: tuple[int, ...]) -> torch.Tensor:
        return self.s0.repeat(*batch_shape, 1)

    def update_masks(self, states: States) -> None:
        states.forward_masks = self._calculate_forward_masks(states)
        states.backward_masks = self._calculate_backward_masks(states)

    def _calculate_forward_masks(self, states: States) -> torch.Tensor:
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

            if n < 0:
                masks[i, 3] = True
                continue

            if phase == 0:
                if n < self.max_nodes:
                    masks[i, 0] = True
                else:
                    masks[i, 3] = True
            elif phase == 1:
                total_edges_to_decide = (n * (n - 1)) // 2
                if edge_cursor < total_edges_to_decide:
                    masks[i, 1] = True
                    masks[i, 2] = True
                else:
                    masks[i, 3] = True

        return masks

    def _calculate_backward_masks(self, states: States) -> torch.Tensor:
        if states.batch_shape:
            batch_size = states.batch_shape[0]
        else:
            batch_size = 1
        masks = torch.zeros((batch_size, self.n_actions - 1), dtype=torch.bool, device=self._device)
        return masks

    def _get_edge_index(self, i: int, j: int) -> int:
        if i >= j:
            raise ValueError(f"Invalid edge: i={i} must be < j={j}")
        return i * self.max_nodes - (i * (i + 1)) // 2 + (j - i - 1)

    def _edge_cursor_to_pair(self, cursor: int, n: int) -> Tuple[int, int]:
        edge_count = 0
        for i in range(n):
            for j in range(i + 1, n):
                if edge_count == cursor:
                    return (i, j)
                edge_count += 1
        raise ValueError(f"Invalid edge cursor {cursor} for n={n}")

    def _state_to_graph(self, state_tensor: torch.Tensor) -> nx.DiGraph:
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
                action = actions_tensor.item() if actions_tensor.dim() == 0 else actions_tensor[0].item()
            else:
                action = actions_tensor[i].item()

            if new_tensor.dim() == 1:
                n = int(new_tensor[0].item())
                phase = int(new_tensor[1].item())
                edge_cursor = int(new_tensor[2].item())
            else:
                n = int(new_tensor[i, 0].item())
                phase = int(new_tensor[i, 1].item())
                edge_cursor = int(new_tensor[i, 2].item())

            if action == 3:
                total_edges = (n * (n - 1)) // 2
                if phase == 1 and edge_cursor >= total_edges:
                    if new_tensor.dim() == 1:
                        new_tensor = self.sf.clone()
                    else:
                        new_tensor[i] = self.sf.clone()
                continue

            if phase == 0:
                if action == 0:
                    new_n = n + 1
                    if new_tensor.dim() == 1:
                        new_tensor[0] = new_n
                        if new_n >= self.max_nodes:
                            new_tensor[1] = 1
                            new_tensor[2] = 0
                    else:
                        new_tensor[i, 0] = new_n
                        if new_n >= self.max_nodes:
                            new_tensor[i, 1] = 1
                            new_tensor[i, 2] = 0

            elif phase == 1:
                total_edges = (n * (n - 1)) // 2
                if edge_cursor < total_edges:
                    i_node, j_node = self._edge_cursor_to_pair(edge_cursor, n)
                    edge_idx = self._get_edge_index(i_node, j_node)

                    if action == 1:
                        if new_tensor.dim() == 1:
                            new_tensor[3 + edge_idx] = 0.0
                        else:
                            new_tensor[i, 3 + edge_idx] = 0.0
                    elif action == 2:
                        if new_tensor.dim() == 1:
                            new_tensor[3 + edge_idx] = 1.0
                        else:
                            new_tensor[i, 3 + edge_idx] = 1.0

                    if new_tensor.dim() == 1:
                        new_tensor[2] = edge_cursor + 1
                    else:
                        new_tensor[i, 2] = edge_cursor + 1

        new_states = self.States(new_tensor)
        return new_states

    def is_action_valid(self, states: States, actions: torch.Tensor,
                       backward: bool = False) -> bool:
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
            phase = int(state[1].item())
            edge_cursor = int(state[2].item())

            if action >= self.n_actions:
                valid[i] = False
                continue

            if n < 0:
                valid[i] = (action == 3)
                continue

            if phase == 0:
                valid[i] = (action == 0)
            elif phase == 1:
                total_edges = (n * (n - 1)) // 2
                if edge_cursor < total_edges:
                    valid[i] = (action == 1 or action == 2)
                else:
                    valid[i] = (action == 3)

        return valid.all().item()

    def backward_step(self, states: States, actions: torch.Tensor) -> States:
        raise NotImplementedError("Backward sampling not supported in staged mode")

    def get_log_reward(self, final_states: States) -> torch.Tensor:
        """
        FIXED REWARD SHAPING:
        log(R) = -beta * S_BD^2

        This creates a proper attractive basin at S_BD=0:
        - Maximum reward at S_BD=0: log(R)=0, R=1
        - Decreasing reward as |S_BD| increases
        - Symmetric penalty for positive/negative deviations
        """
        batch_size = final_states.batch_shape[0] if final_states.batch_shape else 1

        # Get adaptive beta (default to 0.01 if not set)
        beta = getattr(self.proxy, 'beta', 0.01)

        if self.proxy.reward_type == 'bd' and hasattr(self.proxy, 'get_bd_energy_batched'):
            energies = self.proxy.get_bd_energy_batched(
                final_states.tensor,
                self.max_nodes
            )
            # FIXED: Negative quadratic centered at S_BD=0
            # log(R) = -beta * S_BD^2
            # At S_BD=0: log(R)=0, R=1 (maximum)
            # As |S_BD| increases: log(R) becomes more negative, R decreases
            log_rewards = -beta * torch.abs(energies)

            # Clamp to prevent numerical issues
            log_rewards = torch.clamp(log_rewards, min=-50.0, max=0.0)

            return log_rewards
        else:
            rewards = []
            for i in range(batch_size):
                if final_states.tensor.dim() == 1:
                    state = final_states.tensor
                else:
                    state = final_states.tensor[i]

                n = int(state[0].item())

                if n != self.max_nodes:
                    rewards.append(-50.0)  # Large penalty for incomplete
                else:
                    g = self._state_to_graph(state)
                    energy = self.proxy.get_energy(g)
                    # Same fix: -beta * S_BD^2
                    log_r = -beta * torch.abs(energy)
                    rewards.append(max(log_r, -50.0))

            return torch.tensor(rewards, dtype=torch.float, device=self._device)

    def log_reward(self, final_states: States) -> torch.Tensor:
        return self.get_log_reward(final_states)
