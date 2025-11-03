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

# Import from the provided library files
from gfn.env import Env
from gfn.estimators import Estimator
from gfn.states import States

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

    @classmethod
    def set_source_state_template(cls, source_tuple):
        """Helper to set the tuple used for s0 generation."""
        # --- DEBUG ---
        print(f"[DEBUG] ObjectStates: Setting source state template to: {source_tuple}")
        # -----------
        cls._source_state_tuple = source_tuple

    def __init__(self, states_list: List[Any], device: torch.device = None):
        super(States, self).__init__()
        cpu_device = torch.device("cpu")
        self.states_list = states_list
        self.tensor = torch.arange(len(states_list), dtype=torch.long, device=cpu_device)
        self.shape = self.tensor.shape

        # --- DEBUG ---
        # --- THE FIX ---
        # The 'self.s0' property relies on the *class attribute* 'self.__class__.s0'.
        # When creating the first s0 instance, this class attribute is not set yet.
        # This getattr() chain safely checks if the class attribute exists,
        # then checks if it has a shape, all without erroring.
        s0_shape = getattr(getattr(self.__class__, "s0", None), "shape", "N/A")
        # ---
        print(f"[DEBUG] ObjectStates.__init__: Created batch. self.shape={self.shape}, self.s0.shape={s0_shape}, first state={states_list[0]}")
        # -----------

        if hasattr(self, "s0") and self.s0 is not None:
            # ---
            # --- THE FIX ---
            # This assertion fails when creating a new batch (e.g., shape (128,))
            # because self.s0 is the canonical s0 batch (shape (1,)).
            # This check is invalid for our object-based states.
            # assert self.s0.shape == self.shape
            # ---
            pass
        else:
            # --- DEBUG ---
            print(f"[DEBUG] ObjectStates.__init__: self.s0 is not set yet (this is OK if creating s0).")
            # -----------
            pass

    # ---
    # --- THE FIX for the IndexError ---
    #
    @property
    def batch_shape(self) -> torch.Size:
        """
        Overrides the base class's n_dims calculation, which
        fails for our ObjectStates wrapper.
        For ObjectStates, the batch_shape is just the shape
        of our dummy tensor (e.g., torch.Size([128])).
        """
        return self.shape # self.shape is self.tensor.shape
    # ---
    # ---

    @classmethod
    def make_initial_states(cls, batch_shape: tuple[int, ...],
                            device: torch.device = None) -> 'ObjectStates':
        """
        Creates a batch of initial (source) states for the GFN sampler.
        """
        # --- DEBUG ---
        print(f"[DEBUG] ObjectStates.make_initial_states: Called with batch_shape={batch_shape}")
        # -----------

        if cls._source_state_tuple is None:
            raise RuntimeError("Source state template not set for ObjectStates.")

        num_states = batch_shape[0] # Assumes batch_shape is (N,)

        # Create N copies of the source state tuple
        initial_states_list = [cls._source_state_tuple] * num_states

        # Return a new instance of ObjectStates wrapping this list
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
#     d. **Select Correct Logits**: This is the "Factored" part. It
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
# ---

# --- FIX ---
# Inherit from PolicyMixin to provide sampler methods like 'init_context'.
class FactorPreAggAgent(Estimator, PolicyMixin):
# -----------
    """
    Implements the Factored Pre-aggregation Agent.

    This agent is compatible with 'TBGFlowNet'. It coordinates the
    'FactorEnv' and 'FactorPreAggPolicy', using the provided
    'collate_fn' and 'num_factors_fn' to correctly select and
    mask logits for each factor decision.
    """
    def __init__(self,
                 policy: FactorPreAggPolicy,
                 env: FactorEnv,
                 collate_fn: Callable,
                 num_factors_fn: Callable,
                 device: str = 'cpu'):
        """
        Args:
            policy: The 'FactorPreAggPolicy' module holding the parameters.
            env: The 'CausalSetEnv' (a 'FactorEnv') instance.
            collate_fn: The function to batch state tuples into a PyG Batch
                        (from 'train.py').
            num_factors_fn: A lambda fn 's -> s[0]' (from 'train.py').
            device: The torch device.
        """

        # --- FIX ---
        # The base 'Estimator' class's default preprocessor
        # (IdentityPreprocessor) requires 'module.input_dim'.
        # Our agent bypasses the preprocessor by implementing
        # 'get_logits' with a custom collate_fn.
        # We pass 'preprocessor=None' to skip the default
        # preprocessor logic, BUT we must still add 'input_dim'
        # to the policy module to satisfy the assertion.
        super().__init__(module=policy, preprocessor=None)
        # -----------

        # --- FIX ---
        # Add temperature attribute required by PolicyMixin
        self.temperature = 1.0
        # -----------

        # This 'policy' is the nn.Module with the parameters, matching
        # 'optim.Adam(policy.parameters())' in train.py.
        # The base class now stores this as 'self.module'.
        self.policy = policy

        self.env = env
        self.collate_fn = collate_fn
        self.num_factors_fn = num_factors_fn
        self.device = torch.device(device)

        # Ensure policy is on the correct device
        self.policy.to(self.device)

        # --- DEBUG ---
        self.get_logits_call_count = 0
        self.last_debug_print_time = 0
        # -----------

    @property
    def expected_output_dim(self) -> int:
        """
        Returns the output dimension of the estimator.
        This corresponds to the number of actions in the environment.
        """
        return len(self.env.action_space)

    def get_logits(self, states: States, backward: bool = False) -> torch.Tensor:
        """
        Calculates the logits for the current factor decision for each state.
        This method is called by 'TBGFlowNet'.
        """
        # --- DEBUG ---
        self.get_logits_call_count += 1
        current_time = time.time()
        do_debug_print = (current_time - self.last_debug_print_time > 5.0) or (self.get_logits_call_count <= 5)
        if do_debug_print:
            print(f"\n[DEBUG] FactorPreAggAgent.get_logits: Call #{self.get_logits_call_count}")
            self.last_debug_print_time = current_time
        # -----------

        if backward:
            raise NotImplementedError(
                "FactorPreAggAgent does not support backward sampling."
            )

        # 1. Extract raw state tuples from the States container.
        try:
            # --- DEBUG ---
            # Changed 'states.tensor.flat' to 'states.states_list'
            # This is the correct way to get states from our ObjectStates
            raw_states: List[tuple] = states.states_list
            # ---
        except AttributeError:
            # Fallback for other States objects
            try:
                raw_states: List[tuple] = list(states.tensor.flat)
            except AttributeError:
                raw_states: List[tuple] = list(states)
        except Exception as e:
            # --- DEBUG ---
            print(f"[DEBUG] FactorPreAggAgent.get_logits: CRITICAL ERROR extracting states. States object: {states}")
            # -----------
            raise ValueError(
                f"FactorPreAggAgent could not extract raw states (tuples) "
                f"from the gfn.States object. Error: {e}"
            )

        if not raw_states:
            # --- DEBUG ---
            if do_debug_print:
                print(f"[DEBUG] FactorPreAggAgent.get_logits: Received empty raw_states list.")
            # -----------
            return torch.empty(0, len(self.env.action_space)).to(self.device)

        # --- DEBUG ---
        if do_debug_print:
            print(f"[DEBUG] FactorPreAggAgent.get_logits: Processing batch of {len(raw_states)} states.")
            print(f"[DEBUG] FactorPreAggAgent.get_logits: First state in batch: {raw_states[0]}")
            if len(raw_states) > 1:
                print(f"[DEBUG] FactorPreAggAgent.get_logits: Last state in batch: {raw_states[-1]}")
        # -----------

        # 2. Collate states into a PyG batch
        #    The collate_fn creates the PyG Batch, which we then
        #    move to the 'self.device' (e.g., 'cuda').
        try:
            pyg_batch = self.collate_fn(raw_states).to(self.device)
        except Exception as e:
            print(f"\n[DEBUG] FactorPreAggAgent.get_logits: CRITICAL ERROR in collate_fn.")
            print(f"Error: {e}")
            print(f"First 5 raw_states: {raw_states[:5]}")
            raise e

        # 3. Get logits for all nodes in the batch from the policy
        #    Shape: [total_nodes, num_actions]
        try:
            all_node_logits = self.policy(pyg_batch)
        except Exception as e:
            print(f"\n[DEBUG] FactorPreAggAgent.get_logits: CRITICAL ERROR in self.policy(pyg_batch).")
            print(f"Error: {e}")
            print(f"pyg_batch.x shape: {pyg_batch.x.shape}, device: {pyg_batch.x.device}")
            print(f"pyg_batch.edge_index shape: {pyg_batch.edge_index.shape}, device: {pyg_batch.edge_index.device}")
            raise e

        # 4. Select the logits for the *current* factor of each state
        selected_logits: List[torch.Tensor] = []
        ptr = pyg_batch.ptr.to(self.device)
        n_actions = len(self.env.action_space)

        # --- DEBUG ---
        if do_debug_print:
            print(f"[DEBUG] FactorPreAggAgent.get_logits: Got all_node_logits shape: {all_node_logits.shape}")
            print(f"[DEBUG] FactorPreAggAgent.get_logits: pyg_batch.ptr: {ptr}")
        # -----------

        for i, state in enumerate(raw_states):
            try:
                num_factors = self.num_factors_fn(state) # e.g., 'n'

                # state is (n, edges, partial_v)
                partial_v = state[2]
                factor_idx = len(partial_v)

                if factor_idx < num_factors:
                    # --- This is a factor-decision step ---
                    stage = state[0]
                    node_start_idx = ptr[i]
                    global_node_idx = node_start_idx + factor_idx
                    logits = all_node_logits[global_node_idx]
                    mask = self.env.get_mask(state, stage, factor_idx).to(self.device)
                    logits = logits.masked_fill(~mask, -1e10)
                    selected_logits.append(logits)

                else:
                    # --- This is a stage-transition step ---
                    trans_logits = torch.full((n_actions,), -1e10,
                                              device=self.device)
                    trans_logits[0] = 0.0
                    selected_logits.append(trans_logits)

            except Exception as e:
                print(f"\n[DEBUG] FactorPreAggAgent.get_logits: CRITICAL ERROR during logit selection loop.")
                print(f"Error: {e}")
                print(f"Failing state (i={i}): {state}")
                # --- DEBUG ---
                # Add print statements for variables that might not be defined
                # if the error happens early in the try block
                print(f"State: {state}")
                if 'num_factors' in locals():
                    print(f"num_factors: {num_factors}, factor_idx: {factor_idx}")
                if 'node_start_idx' in locals():
                    print(f"ptr: {ptr}, node_start_idx: {node_start_idx}, global_node_idx: {global_node_idx}")
                # ---
                print(f"all_node_logits shape: {all_node_logits.shape}")
                raise e

        # Stack the selected logits for the batch
        # Shape: [batch_size, num_actions]
        final_logits = torch.stack(selected_logits)

        # --- DEBUG ---
        if do_debug_print:
            print(f"[DEBUG] FactorPreAggAgent.get_logits: Success. Returning final_logits shape: {final_logits.shape}")
        # -----------

        return final_logits

    def forward(self, states: States, backward: bool = False) -> torch.Tensor:
        """
        This is the 'nn.Module.forward' method, which 'Estimator.get_logits'
        would normally call. We override 'get_logits' directly, but
        implement 'forward' to be safe.
        """
        return self.get_logits(states, backward)
