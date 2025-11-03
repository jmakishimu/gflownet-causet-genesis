# custom_gfn.py
#
# This file contains the implementations of FactorEnv, FactorPreAggPolicy,
# and FactorPreAggAgent, which are missing from the provided torchgfn library.
# These implementations are based on the principled requirements of GFlowNets
# with factored action spaces and are designed to be compatible with the
# causet_env.py, causet_policy.py, and train.py files.
#
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Batch
from abc import ABC, abstractmethod
from typing import Callable, List, Any
import time  # --- DEBUG ---
# --- FIX ---
# Import the distribution module for the new method
import torch.distributions as dist
# -----------

# Import from the provided library files
from gfn.env import Env
from gfn.estimators import Estimator
from gfn.states import States
# ---
# --- FIX for Actions ---
from gfn.actions import Actions
# ---
# ---

# --- FIX ---
# Import PolicyMixin from 'gfn.estimators' instead of 'gfn.modules'
from gfn.estimators import PolicyMixin
# -----------


# ---
# Theory and Justification for FactorEnv
#
# 1.  **Purpose**: 'FactorEnv' represents an environment where a single "stage"
#     transition (e.g., adding the (n+1)-th node) is broken down into a
#     sequence of "factor" decisions (e.g., deciding each of the 'n'
#     potential new edges).
# 2.  **Inheritance**: It must inherit from 'gfn.env.Env' so that the
#     'TBGFlowNet' algorithm can use it.
# 3.  **The __init__ Problem**: The base 'gfn.env.Env' class requires
#     tensor-based states for its '__init__'. Your 'CausalSetEnv'
#     uses tuple-based states and a simple 'super().__init__()' call.
# 4.  **Solution**: This 'FactorEnv' acts as a "bridge." It inherits from 'Env'
#     but provides its own '__init__' that does *not* call the parent 'Env'
#     '__init__'. This satisfies the 'isinstance(env, Env)' check while
#     bypassing the tensor-state requirement. Your 'CausalSetEnv'
#     then inherits this behavior.
# 5.  **Abstract Methods**: It defines 'get_num_factors' and 'get_mask' as
#     abstract, as these are the core methods a factored environment must
#     provide, and which your 'CausalSetEnv' does provide.
# 6.  **Env's Abstract Methods**: The base 'Env' also has abstract methods
#     'step', 'backward_step', and 'is_action_valid'. 'CausalSetEnv'
#     implements 'step'. We provide stubs for the other two, as your
#     training loop relies on 'get_mask' (via the Agent) instead of
#     'is_action_valid', and does not use 'backward_step'.
# ---

class FactorEnv(Env, ABC):
    """
    An abstract base class for Environments with Factored Action Spaces.

    This class acts as a bridge to allow tuple-based states (like in
    CausalSetEnv) to be used with the 'torchgfn' library, which expects
    'gfn.env.Env' instances.
    """
    def __init__(self):
        """
        This __init__ intentionally does not call super().__init__ from
        gfn.env.Env. This is to avoid the base class's requirement
        for tensor-based s0, sf, etc., which is incompatible with the
        tuple-based states used in CausalSetEnv.

        Subclasses (like CausalSetEnv) are responsible for setting
        'self.source' (as s0) and other necessary attributes.
        """
        pass # Intentionally blank.

    @abstractmethod
    def get_num_factors(self, state: Any) -> int:
        """
        Returns the number of factor decisions for the current stage.
        """
        pass

    @abstractmethod
    def get_mask(self, state: Any, stage: int, factor_idx: int) -> torch.Tensor:
        """
        Returns a mask for valid actions for a specific factor decision.
        """
        pass

    # --- Implement abstract methods from gfn.env.Env ---

    @abstractmethod
    def step(self, state: Any, action: Any) -> tuple[Any, Any, bool]:
        """
        This method is implemented by the subclass (e.g., CausalSetEnv).

        Returns: (next_state, action_taken, is_done)
        """
        pass

    # ---
    # --- FIX for Vectorized Env ---
    #
    # The torchgfn library (as per your traceback) calls `env._step`
    # (the non-vectorized version) which iterates and calls `env.step`
    # (the single-state version).
    #
    # We must rename CausalSetEnv's vectorized step to `_step`
    # and provide a single-state `step` to satisfy the API.
    #
    # To fix this, we implement the *single-state* `step` here,
    # which will be called by the library's default `_step`.
    # We also add an `_step` abstract method for CausalSetEnv
    # to optionally implement for vectorization.

    @abstractmethod
    def _step_single(self, state: Any, action: Any) -> tuple[Any, Any, bool]:
        """
        The abstract single-state step logic that CausalSetEnv must implement.
        """
        pass

    def step(self, state: Any, action: Any) -> tuple[Any, Any, bool]:
        """
        This is the single-state step function that the base library's
        `_step` will call. It delegates to the child's implementation.
        """
        return self._step_single(state, action)

    # ---
    # --- NEW (Vectorized) _step ---
    #
    # Per the `torchgfn` docs, if we want a vectorized environment,
    # we must override `_step`, not `step`.

    def _step(self, states: "States", actions: "Actions") -> "States":
        """
        This is the *vectorized* step function.
        We provide a default implementation that calls the single-state `step`.
        A child class (like CausalSetEnv) can override this
        for a fully vectorized implementation.
        """
        # This is the default non-vectorized implementation from gfn.env.Env
        # We include it here so CausalSetEnv can override it.
        new_states = states.copy()
        dones = states.is_sink_state

        for i, state in enumerate(states):
            if not dones[i]:
                action = actions[i]
                new_state, action_taken, done = self.step(state, action)
                new_states[i] = new_state
                dones[i] = done

        return new_states
    # ---
    # --- END FIX ---
    # ---

    def backward_step(self, states: Any, actions: Any) -> Any:
        """
        Backward sampling is not implemented for this factored environment.
        """
        raise NotImplementedError("Factored environments are forward-only")

    def is_action_valid(self, states: Any, actions: Any,
                        backward: bool = False) -> bool:
        """
        This check is superseded by the Agent's use of 'get_mask'.
        We return True to satisfy the Env API. The true constraint
        satisfaction happens in 'FactorPreAggAgent.get_logits'.
        """
        if backward:
            return False
        return True

# ---
# --- NEW CLASS ---
# Implementation for ObjectStates, which was missing.
# ---

class ObjectStates(States):
    """
    A 'States' container for states that are Python objects (like tuples).
    It stores them in a Python list and provides a dummy tensor for API compatibility.
    """

    # We will store the *template* source state here once the environment defines it.
    _source_state_tuple = None

    # ---
    # --- FIX for "weird" sampler bug ---
    #
    # These class attributes will be "patched" by the CausalSetEnv
    # instance so that `is_sink_state` can correctly identify
    # terminal graph states.
    env_max_nodes = None
    sf_value = -1

    @classmethod
    def patch_env_config(cls, env_max_nodes: int, sf_value: Any):
        """
        Called by the environment to give this class
        access to environment-specific terminal conditions.
        """
        # print(f"[DEBUG] ObjectStates: Patching config. max_nodes={env_max_nodes}, sf_value={sf_value}")
        cls.env_max_nodes = env_max_nodes
        cls.sf_value = sf_value
    # ---
    # --- END OF FIX ---
    # ---

    @classmethod
    def set_source_state_template(cls, source_tuple):
        """Helper to set the tuple used for s0 generation."""
        # --- DEBUG ---
        # print(f"[DEBUG] ObjectStates: Setting source state template to: {source_tuple}")
        # -----------
        cls._source_state_tuple = source_tuple

    def __init__(self, states_list: List[Any], device: torch.device = None):
        # ---
        # --- THIS IS THE FIX (Part 1) ---
        # This robust __init__ handles all cases:
        # 1. The correct case (a list of tuples)
        # 2. The `sf` case (a list with one int, e.g., [-1])
        # 3. The BUGGY base.copy() case (a torch.Tensor)
        # ---
        if isinstance(states_list, torch.Tensor):
            # This happens if base `States.copy()` is called.
            # It passes `self.tensor.clone()`, which is `[0, 1, 2, ...]`.
            # We convert it to a Python list.
            processed_states_list = states_list.tolist()
        elif not isinstance(states_list, list):
            # This handles the `sf` case where `states_list` is `[-1]`.
            try:
                processed_states_list = list(states_list)
            except TypeError:
                # `list(-1)` fails, so we catch it.
                processed_states_list = [states_list]
        else:
            # This is the "correct" path (e.g., from env.step or make_initial_states)
            processed_states_list = states_list
        # ---

        super(States, self).__init__()

        # --- DEVICE FIX ---
        # Use the provided device, or default to CPU.
        # This ensures self.tensor (and self.device) are correct.
        self._device = device if device is not None else torch.device("cpu")

        # ---
        # --- THIS IS THE FIX (Part 2) ---
        # Use the *processed* list, not the original `states_list`.
        self.states_list = processed_states_list
        # ---

        # ---
        # --- THIS IS THE FIX for the `AssertionError` ---
        # The Trajectories container asserts `len(batch_shape) == 2`.
        # We must make our dummy tensor 2D, e.g., [batch_size, 1].
        self.tensor = torch.arange(len(self.states_list), dtype=torch.long, device=self._device).unsqueeze(-1)
        # ---

        self.shape = self.tensor.shape

        # --- DEBUG ---
        # --- THE FIX ---
        # The 'self.s0' property relies on the *class attribute* 'self.__class__.s0'.
        # When creating the first s0 instance, this class attribute is not set yet.
        # This getattr() chain safely checks if the class attribute exists,
        # then checks if it has a shape, all without erroring.
        s0_shape = getattr(getattr(self.__class__, "s0", None), "shape", "N/A")
        # ---
        # --- THIS IS THE FIX (Part 3) ---
        # Use self.states_list[0] for the debug print, which is now safe.
        first_state_debug = self.states_list[0] if self.states_list else "N/A"
        # This print statement will now show `first state=0` when the bug occurs
        # if the copy() method is not overridden.
        # print(f"[DEBUG] ObjectStates.__init__: Created batch. self.shape={self.shape}, self.s0.shape={s0_shape}, first state={first_state_debug}")
        # -----------

        if hasattr(self, "s0") and self.s0 is not None:
            # ---
            # --- THE FIX ---
            # This assertion fails when creating a new batch (e.g., shape (128,1))
            # because self.s0 is the canonical s0 batch (shape (1,1)).
            # This check is invalid for our object-based states.
            # We can check the number of dimensions instead.
            assert self.s0.shape[1:] == self.shape[1:]
            # ---
            pass
        else:
            # --- DEBUG ---
            # print(f"[DEBUG] ObjectStates.__init__: self.s0 is not set yet (this is OK if creating s0).")
            # -----------
            pass

    # ---
    # --- THIS IS THE CRITICAL FIX (Part 4) ---
    #
    def copy(self) -> "ObjectStates":
        """
        Overrides the base 'States.copy()' method.
        The base method copies 'self.tensor', which is wrong for ObjectStates.
        We must copy 'self.states_list'.
        """
        # --- DEBUG ---
        # print(f"[DEBUG] ObjectStates.copy: Copying states_list (len={len(self.states_list)})")
        # ---
        # Perform a shallow copy of the list of tuples.
        # This is safe because the tuples themselves are immutable.
        new_states_list = list(self.states_list)

        # Return a *new* instance using our *correct* __init__.
        # This will call __init__ with a valid list of tuples,
        # and the "first state=0" log will NOT appear.
        return self.__class__(new_states_list, device=self.device)
    # ---
    # --- END OF CRITICAL FIX ---
    # ---


    # ---
    # --- THE FIX for the IndexError ---
    #

    # ---
    # --- THIS IS THE FIX for `len(batch_shape) == 2` ---
    # We override the n_dims class attribute to 0.
    n_dims = 0

    @property
    def batch_shape(self) -> torch.Size:
        """
        The base class defines batch_shape as self.shape[:-self.n_dims].
        With n_dims=0, this would be self.shape[:], which is (B, 1).
        This seems to be what the Trajectories class wants.
        However, the *sampler* (samplers.py line 245) does:
        `self.env.States.make_initial_states(batch_shape=(n_samples, *self.env.state_shape)`
        This implies `env.state_shape` should be `(1,)` and
        the *base* `batch_shape` (which is `self.shape[:-1]`)
        should be used.

        This is a deep contradiction. Let's force it.
        We will return self.shape `(B, 1)` to satisfy Trajectories,
        and modify `make_initial_states` to handle it.
        """
        return self.shape
    # ---
    # ---

    # ---
    # --- FIX for 'is_sink_state' (The "weird" sampler bug) ---
    #
    @property
    def is_sink_state(self) -> torch.Tensor:
        """
        Checks which states in the batch are the sink state.
        This property is required by the GFN sampler.

        This implementation correctly identifies *both* the
        canonical sink state (e.g., -1) and environment-specific
        terminal states (e.g., graphs with n == max_nodes).
        """
        if self.__class__.env_max_nodes is None:
            raise RuntimeError(
                "ObjectStates.env_max_nodes was not patched by the environment. "
                "Ensure CausalSetEnv.patch_env_config() is called."
            )

        # Get the patched values from the class
        max_n = self.__class__.env_max_nodes
        sink_val = self.__class__.sf_value

        is_sink = []
        for s in self.states_list:
            if s == sink_val:
                # Case 1: It's the canonical sink state (e.g., -1)
                is_sink.append(True)
            elif isinstance(s, tuple) and len(s) == 3:
                # Case 2: It's a graph state. Check if terminal.
                n = s[0] # Get node count
                is_sink.append(n == max_n)
            else:
                # Case 3: It's some other state (e.g., s0)
                is_sink.append(False)

        # --- DEVICE FIX ---
        # Return as a boolean tensor on the correct device.
        return torch.tensor(is_sink, dtype=torch.bool, device=self.tensor.device)
        # ---
    # ---
    # --- END OF FIX ---
    # ---

    # ---
    # --- THIS IS THE CRITICAL FIX (Part 5) ---
    #
    def __setitem__(self, index: Any, value: "ObjectStates"):
        """
        Handles setting a slice of the states.
        This is required by 'env._step' (e.g., new_states[not_dones] = ...).
        """
        # --- DEBUG ---
        # print(f"[DEBUG] ObjectStates.__setitem__: Setting items. Index type: {type(index)}")
        # ---

        # --- FIX ---
        # The base non-vectorized `_step` calls __setitem__ with a
        # *single* state (e.g., a tuple), not an ObjectStates object.
        # We must handle this.
        if isinstance(value, ObjectStates):
            new_states_to_set = value.states_list
        else:
            # It's a single state (e.g., a tuple)
            new_states_to_set = [value]
        # ---

        if isinstance(index, torch.Tensor) and index.dtype == torch.bool:
            # This is the expected case: new_states[not_dones]
            index_list = index.cpu().numpy()
            value_iter = iter(new_states_to_set)
            for i in range(len(index_list)):
                if index_list[i]:
                    try:
                        # This correctly modifies the `self.states_list`
                        self.states_list[i] = next(value_iter)
                    except StopIteration:
                        raise IndexError("Not enough values to set in ObjectStates __setitem__")
        else:
            # Handle other index types (like a single int from non-vectorized loop)
            try:
                self.states_list[index] = new_states_to_set[0]
            except Exception as e:
                raise TypeError(f"ObjectStates __setitem__ encountered an error. Index: {index}, Value: {value}. Error: {e}")

    # ---
    # --- END OF CRITICAL FIX ---
    # ---

    # ---
    # --- THIS IS THE FIX ---
    #
    def __getitem__(self, index: Any) -> Any:
        """
        Handles slicing and indexing of the ObjectStates batch.
        This is required by the sampler to filter active states (with a bool tensor)
        and to iterate (with an int).
        """

        if isinstance(index, int):
            # Handle single item access (e.g., states[0])
            # This is called by the base Env.step() when iterating.
            return self.states_list[index] # Returns the raw tuple

        if isinstance(index, torch.Tensor):
            if index.dtype == torch.bool:
                # Handle boolean mask (e.g., states[~dones])
                index_list = index.cpu().numpy()
            else:
                # Handle integer tensor slicing (e.g., states[torch.tensor([0, 2])])
                filtered_states = [self.states_list[i] for i in index.cpu().numpy()]
                return self.__class__(filtered_states, device=self.device)
        elif isinstance(index, slice):
            # Handle slice access (e.g., states[1:3])
            return self.__class__(self.states_list[index], device=self.device)
        else:
            # Assume boolean list/numpy array
            index_list = index

        # Handle boolean list/array filtering (e.g., states[~dones])
        filtered_states = [
            state for i, state in enumerate(self.states_list) if index_list[i]
        ]

        # Return a *new* ObjectStates object with the filtered list.
        return self.__class__(filtered_states, device=self.device)
    # ---
    # --- END OF FIX ---
    # ---

    @classmethod
    def make_initial_states(cls, batch_shape: tuple[int, ...],
                            device: torch.device = None) -> 'ObjectStates':
        """
        Creates a batch of initial (source) states for the GFN sampler.
        """
        # --- DEBUG ---
        # print(f"[DEBUG] ObjectStates.make_initial_states: Called with batch_shape={batch_shape}")
        # -----------

        if cls._source_state_tuple is None:
            raise RuntimeError("Source state template not set for ObjectStates.")

        # ---
        # --- AssertionError FIX ---
        # The sampler passes a `batch_shape` of `(n_samples, 1)`.
        # We must extract the number of states from `batch_shape[0]`.
        num_states = batch_shape[0] # Assumes batch_shape is (N, 1) or (N,)
        # ---

        # Create N copies of the source state tuple
        initial_states_list = [cls._source_state_tuple] * num_states

        # Return a new instance of ObjectStates wrapping this list
        # This device is passed from the sampler (e.g., cuda:0)
        return cls(initial_states_list, device=device)
# --- END NEW CLASS ---
# ---


# ---
# Theory and Justification for FactorPreAggPolicy
#
# 1.  **Purpose**: This module acts as the GFlowNet policy network.
# 2.  **Inheritance**: It's a 'torch.nn.Module', as it holds the trainable
#     parameters that the 'train.py' optimizer will update.
# 3.  **Structure**: As described in 'causet_policy.py', this network
#     is composed of two parts:
#     a) 'self.graph_network': The GNN (e.g., 'GNNPolicy') that
#        processes the graph and outputs node embeddings.
#     b) 'self.mlp': A simple Linear layer that projects the node
#        embeddings into logits for the factor decisions.
# 4.  **Forward Pass**: The 'forward' method takes a 'pyg_batch' (created
#     by the Agent's 'collate_fn'), passes it through the GNN to get
#     node embeddings, and then through the MLP to get logits for *all*
#     nodes in the batch. The Agent is responsible for selecting the
#     correct logits for the *current* factor.
# ---

class FactorPreAggPolicy(nn.Module):
    """
    Implements the Factored Policy network.

    This module holds the GNN backbone and the final MLP head, as
    described in 'causet_policy.py'.
    """
    def __init__(self,
                 graph_network: nn.Module,
                 input_dim: int,
                 node_dim: int,
                 num_actions: int):
        """
        Args:
            graph_network: The GNN backbone (e.g., GNNPolicy).
            input_dim: The input feature dimension (e.g., 5).
                       (Required by Estimator base class).
            node_dim: The output dimension of the GNN.
            num_actions: The number of choices for each factor (e.g., 2).
        """
        super().__init__()
        self.graph_network = graph_network

        # --- FIX ---
        # Added 'self.input_dim' to satisfy the assertion in the
        # base 'Estimator' class __init__.
        self.input_dim = input_dim
        # -----------

        # The "separate MLP" mentioned in causet_policy.py
        self.mlp = nn.Linear(node_dim, num_actions)

    def forward(self, pyg_batch: Batch) -> torch.Tensor:
        """
        Processes a PyG batch and returns logits for all nodes.

        Args:
            pyg_batch: A 'torch_geometric.data.Batch' object from
                       the agent's collate_fn.

        Returns:
            A tensor of shape [total_nodes_in_batch, num_actions]
            containing the logits for each node.
        """
        # Get node embeddings from the GNN
        # Shape: [total_nodes_in_batch, node_dim]
        node_embeddings = self.graph_network(pyg_batch)

        # Pass embeddings through MLP to get logits for each node
        # Shape: [total_nodes_in_batch, num_actions]
        factor_logits = self.mlp(node_embeddings)

        return factor_logits

# ---
# --- FIX for 'AssertionError' (shape mismatch) ---
#
class _CategoricalWrapper(dist.Categorical):
    """
    A wrapper for the Categorical distribution to fix a shape mismatch.
    The default 'dist.sample()' returns [batch_size], but the 'gfn.Actions'
    class expects [batch_size, 1]. This wrapper fixes that.

    (Note: This wrapper inherits __init__ from dist.Categorical, so
    it correctly accepts 'probs=...' or 'logits=...'.)
    """
    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        # Call the original sample method
        actions_tensor = super().sample(sample_shape)

        # Add the missing dimension: [batch_size] -> [batch_size, 1]
        return actions_tensor.unsqueeze(-1)

# ---
# --- END OF FIX ---
# ---

# ---
# Theory and Justification for FactorPreAggAgent
#
# 1.  **Purpose**: This class is the "Agent." It bridges the 'FactorEnv'
#     and the 'FactorPreAggPolicy'.
# 2.  **Inheritance**: It inherits from 'gfn.estimators.Estimator'
#     so that 'TBGFlowNet' can use it as a 'pf' or 'pb' module.
# 3.  **The 'get_logits' Method**: This is the core of the implementation.
#     The 'TBGFlowNet' sampler will call this method with a batch of
#     'States'. This method must return the correct logits
#     '([batch_size, num_actions])' for the *specific factor* being
#     decided at each state.
# 4.  **Logic**:
#     a. **Unpack States**: It first extracts the raw Python state tuples
#        (e.g., '(n, e, partial_v)') from the 'gfn.states.States'
#        container. (This works because CausalSetEnv sets self.States
#        to ObjectStates, which is a subclass of States).
#     b. **Collate**: It uses the 'collate_fn' (from 'train.py') to
#        convert this list of tuples into a single 'pyg_batch'.
#     c. **Get All Logits**: It calls 'self.policy(pyg_batch)' to get the
#        logits for *all nodes in the entire batch*.
#     d. **Select Correct Logits**: This is "Factored" part. It
#        iterates through the states in the batch. For each state:
#        i.   It finds the current 'factor_idx' (from 'len(partial_v)').
#        ii.  It uses the 'pyg_batch.ptr' to find the *global node index*
#             corresponding to that 'factor_idx' for that specific graph.
#        iii. It selects the correct logits: 'all_node_logits[global_node_idx]'.
#     e. **Apply Mask**: It then calls 'self.env.get_mask(...)' for that
#        'state' and 'factor_idx', applying the mask to the selected logits.
#        This enforces the transitivity constraints.
#     f. **Stage Transitions**: It handles states where all factors are
#        complete ('factor_idx == num_factors'), such as 's0' or a
#        completed state '(n, e, (a_0,...,a_{n-1}))'. For these, it
#        returns fixed logits forcing "action 0", which triggers the
#        'step' method's stage-transition logic.
# 5.  **Epsilon-Greedy**: This is implemented in `to_probability_distribution`.
#     It mixes the `softmax(logits)` from the policy with a uniform
#     distribution over valid actions, weighted by `self.epsilon`.
# ---

# --- FIX ---
# Inherit from PolicyMixin to provide sampler methods like 'init_context'.
class FactorPreAggAgent(Estimator, PolicyMixin):
    """
    Implements the Factored Pre-aggregation Agent with exploration.
    """
    def __init__(self,
                 policy: FactorPreAggPolicy,
                 env: FactorEnv,
                 collate_fn: Callable,
                 num_factors_fn: Callable,
                 device: str = 'cpu',
                 temperature: float = 1.0,
                 epsilon: float = 0.1):
        """
        Args:
            policy: The 'FactorPreAggPolicy' module holding the parameters.
            env: The 'CausalSetEnv' (a 'FactorEnv') instance.
            collate_fn: The function to batch state tuples into a PyG Batch
                        (from 'train.py').
            num_factors_fn: A lambda fn 's -> s[0]' (from 'train.py').
            device: The torch device.
            temperature: Softmax temperature for sampling.
            epsilon: Probability of random action (epsilon-greedy).
        """
        super().__init__(module=policy, preprocessor=None)

        self.policy = policy
        self.env = env
        self.collate_fn = collate_fn
        self.num_factors_fn = num_factors_fn
        self.device = torch.device(device)

        # Exploration parameters
        self.temperature = temperature
        self.epsilon = epsilon
        self._training_mode = True  # Track if we're in training

        # Ensure policy is on the correct device
        self.policy.to(self.device)

        # Debug counters
        self.get_logits_call_count = 0
        self.last_debug_print_time = 0

    def set_epsilon(self, epsilon: float):
        """Update epsilon for exploration annealing."""
        self.epsilon = epsilon

    def set_temperature(self, temperature: float):
        """Update temperature for annealing."""
        self.temperature = temperature

    @property
    def expected_output_dim(self) -> int:
        return len(self.env.action_space)

    def get_logits(self, states: States, backward: bool = False) -> torch.Tensor:
        """
        Calculates the logits for the current factor decision for each state.
        """
        self.get_logits_call_count += 1
        current_time = time.time()
        do_debug_print = (current_time - self.last_debug_print_time > 5.0) or (self.get_logits_call_count <= 5)
        self.last_debug_print_time = current_time

        if backward:
            raise NotImplementedError("FactorPreAggAgent does not support backward sampling.")

        # Extract raw state tuples
        try:
            raw_states: List[tuple] = states.states_list
        except AttributeError:
            try:
                raw_states: List[tuple] = list(states.tensor.flat)
            except AttributeError:
                raw_states: List[tuple] = list(states)
        except Exception as e:
            raise ValueError(f"FactorPreAggAgent could not extract raw states. Error: {e}")

        if not raw_states:
            return torch.empty(0, len(self.env.action_space)).to(self.device)

        # Collate states into PyG batch
        try:
            pyg_batch = self.collate_fn(raw_states).to(self.device)
        except Exception as e:
            print(f"\n[ERROR] collate_fn failed: {e}")
            raise e

        # Get logits from policy
        try:
            all_node_logits = self.policy(pyg_batch)
        except Exception as e:
            print(f"\n[ERROR] policy forward failed: {e}")
            raise e

        # Apply temperature scaling
        all_node_logits = all_node_logits / self.temperature

        # Select logits for current factor
        selected_logits: List[torch.Tensor] = []
        ptr = pyg_batch.ptr.to(self.device)
        n_actions = len(self.env.action_space)

        for i, state in enumerate(raw_states):
            try:
                n, edges, partial_v = state
                num_factors = self.num_factors_fn(state)
                factor_idx = len(partial_v)

                # Terminal state check
                if n == self.env.max_nodes:
                    trans_logits = torch.full((n_actions,), -1e10, device=self.device)
                    trans_logits[0] = 0.0
                    selected_logits.append(trans_logits)

                elif factor_idx < num_factors:
                    # Factor decision step
                    stage = n
                    node_start_idx = ptr[i]
                    global_node_idx = node_start_idx + factor_idx
                    logits = all_node_logits[global_node_idx]

                    # Apply mask
                    mask = self.env.get_mask(state, stage, factor_idx).to(self.device)
                    logits = logits.masked_fill(~mask, -1e10)

                    selected_logits.append(logits)

                else:
                    # Stage transition step
                    trans_logits = torch.full((n_actions,), -1e10, device=self.device)
                    trans_logits[0] = 0.0
                    selected_logits.append(trans_logits)

            except Exception as e:
                print(f"\n[ERROR] Logit selection failed for state {i}: {state}")
                print(f"Error: {e}")
                raise e

        final_logits = torch.stack(selected_logits)
        return final_logits

    def forward(self, states: States, backward: bool = False) -> torch.Tensor:
        return self.get_logits(states, backward)

    def to_probability_distribution(self, states: States, context: torch.Tensor) -> dist.Distribution:
        """
        Creates distribution with epsilon-greedy exploration during training.
        """
        logits = context

        # During training, add epsilon-greedy exploration
        if self._training_mode and self.epsilon > 0:
            batch_size = logits.shape[0]

            # Create uniform logits for exploration
            uniform_logits = torch.zeros_like(logits)

            # Mix policy and uniform based on epsilon
            # With probability epsilon, use uniform; otherwise use policy
            mixed_logits = torch.log(
                self.epsilon * torch.softmax(uniform_logits, dim=-1) +
                (1 - self.epsilon) * torch.softmax(logits, dim=-1)
            )

            return _CategoricalWrapper(logits=mixed_logits)
        else:
            return _CategoricalWrapper(logits=logits)

    def train(self, mode: bool = True):
        """Override train mode to track exploration."""
        self._training_mode = mode
        self.policy.train(mode)
        return self

    def eval(self):
        """Override eval mode to disable exploration."""
        self._training_mode = False
        self.policy.eval()
        return self
# ---
# --- FIX for `AssertionError: self.actions.batch_shape` ---
#
class CustomActions(Actions):
    """
    A custom Actions container to fix a batch_shape assertion.
    The Trajectories container requires batch_shape to be 2D.
    """
    @property
    def batch_shape(self) -> torch.Size:
        """
        Returns the batch shape for actions.
        The tensor shape is [batch_size, action_dim] where action_dim = 1.
        We need to return just [batch_size] to match what Trajectories expects.
        """
        # The base Actions class expects batch_shape to exclude the action dimensions
        # For a tensor of shape [B, 1], batch_shape should be [B]
        if len(self.tensor.shape) >= 2:
            return self.tensor.shape[:1]  # Return just [batch_size]
        else:
            # Fallback for 1D tensor
            return self.tensor.shape
# ---
# --- END OF FIX ---
# ---
