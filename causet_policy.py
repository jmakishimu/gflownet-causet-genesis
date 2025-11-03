#
# gflownet-causet-genesis/causet_policy.py
#
import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data, Batch
from torch_geometric.utils import from_networkx
import networkx as nx

class GNNPolicy(nn.Module):
    """
    Implements the GNN Policy (Phase 2).

    This GNN processes the *current* causet `g` (of size n) and outputs
    `n` node embeddings. A separate MLP (in the agent) will use these
    embeddings to produce the `n` logits for the factors of stage `n+1`.
    """
    def __init__(self, node_feature_dim=5, hidden_dim=128, out_dim=128):
        super().__init__()
        self.node_feature_dim = node_feature_dim

        # Input MLP to project rich features
        self.node_embed = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # GNN layers (SOTA as requested)
        self.gnn1 = pyg_nn.GATv2Conv(hidden_dim, hidden_dim, heads=4)
        self.gnn2 = pyg_nn.GATv2Conv(hidden_dim * 4, out_dim, heads=1)

    def forward(self, pyg_batch: Batch):
        """
        Processes a PyG Batch object and returns node embeddings.

        Input: A PyG Batch object from the collate_fn.
        Output: A tensor of node embeddings, shape [total_nodes, out_dim]
        """
        x, edge_index = pyg_batch.x, pyg_batch.edge_index

        if x.shape[0] == 0: # Handle empty graph
            return torch.empty(0, self.gnn2.out_channels).to(x.device)

        x = self.node_embed(x)
        x = self.gnn1(x, edge_index).relu()
        x = self.gnn2(x, edge_index)

        return x
