"""
Core building blocks for PharmHGT.

Includes:
  - MultiHeadAttention: standard scaled dot-product attention
  - AttentionConv: graph attention message passing (homo edges)
  - HeteroGNNLayer: single layer of heterogeneous message passing
  - GraphGRU: attention-based GRU readout for graph-level representation
  - MVMP: multi-view message passing module
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter, softmax


# ===========================================================================
# Standard multi-head attention (for GRU readout)
# ===========================================================================
class MultiHeadAttention(nn.Module):
    def __init__(self, hid_dim, num_heads=4, dropout=0.1):
        super().__init__()
        assert hid_dim % num_heads == 0
        self.hid_dim = hid_dim
        self.num_heads = num_heads
        self.head_dim = hid_dim // num_heads

        self.q_proj = nn.Linear(hid_dim, hid_dim)
        self.k_proj = nn.Linear(hid_dim, hid_dim)
        self.v_proj = nn.Linear(hid_dim, hid_dim)
        self.out_proj = nn.Linear(hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        B, Lq, _ = query.shape
        _, Lk, _ = key.shape

        Q = self.q_proj(query).view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ V).transpose(1, 2).contiguous().view(B, Lq, self.hid_dim)
        return self.out_proj(out), attn


# ===========================================================================
# Attention-based message passing for homogeneous edge types
# ===========================================================================
class AttentionConv(MessagePassing):
    """Graph attention message passing with residual connection.

    h_i' = h_i + W_out( sum_j attn_ij * W_v h_j )
    attn_ij = softmax_j( (W_q h_i . W_k h_j) / sqrt(d_k) )
    """

    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.1):
        super().__init__(aggr='add', node_dim=0)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads

        self.q = nn.Linear(in_dim, out_dim)
        self.k = nn.Linear(in_dim, out_dim)
        self.v = nn.Linear(in_dim, out_dim)
        self.out = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j, index):
        q = self.q(x_i).view(-1, self.num_heads, self.head_dim)
        k = self.k(x_j).view(-1, self.num_heads, self.head_dim)
        v = self.v(x_j).view(-1, self.num_heads, self.head_dim)

        attn = (q * k).sum(dim=-1) / math.sqrt(self.head_dim)
        attn = softmax(attn, index, dim=0)
        attn = self.dropout(attn)

        return (attn.unsqueeze(-1) * v).view(-1, self.num_heads * self.head_dim)

    def update(self, aggr_out, x):
        return x + self.out(aggr_out)


# ===========================================================================
# Heterogeneous graph layer (one round of message passing over all edge types)
# ===========================================================================
class HeteroGNNLayer(nn.Module):
    """Single layer of heterogeneous message passing.

    - Homo edges (bond, react): AttentionConv
    - Cross-type edges (junc): scatter mean
    """

    def __init__(self, hid_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.bond_conv = AttentionConv(hid_dim, hid_dim, num_heads, dropout)
        self.react_conv = AttentionConv(hid_dim, hid_dim, num_heads, dropout)

        self.atom_lin = nn.Linear(hid_dim, hid_dim)
        self.pharm_lin = nn.Linear(hid_dim, hid_dim)
        self.norm = nn.LayerNorm(hid_dim)

    def forward(self, x_dict, edge_index_dict):
        atom_x = x_dict['atom']
        pharm_x = x_dict['pharm']

        n_atom = atom_x.size(0)
        n_pharm = pharm_x.size(0)
        device = atom_x.device

        # --- Homogeneous attention ---
        if edge_index_dict[('atom', 'bond', 'atom')].size(1) > 0:
            new_atom = self.bond_conv(atom_x, edge_index_dict[('atom', 'bond', 'atom')])
        else:
            new_atom = atom_x

        if n_pharm > 0 and edge_index_dict[('pharm', 'react', 'pharm')].size(1) > 0:
            new_pharm = self.react_conv(pharm_x, edge_index_dict[('pharm', 'react', 'pharm')])
        else:
            new_pharm = pharm_x

        # --- Cross-type junction via scatter ---
        ap_idx = edge_index_dict[('atom', 'junc', 'pharm')]
        pa_idx = edge_index_dict[('pharm', 'junc', 'atom')]

        atom_cross = torch.zeros_like(new_atom)
        pharm_cross = torch.zeros_like(new_pharm)

        if ap_idx.size(1) > 0:
            src_atom = atom_x[ap_idx[0]]
            pharm_agg = scatter(src_atom, ap_idx[1], dim=0, dim_size=n_pharm, reduce='mean')
            pharm_cross = pharm_agg

        if pa_idx.size(1) > 0:
            src_pharm = pharm_x[pa_idx[0]]
            atom_agg = scatter(src_pharm, pa_idx[1], dim=0, dim_size=n_atom, reduce='mean')
            atom_cross = atom_agg

        atom_out = self.norm(new_atom + self.atom_lin(atom_cross))
        pharm_out = self.norm(new_pharm + self.pharm_lin(pharm_cross))

        return {'atom': atom_out, 'pharm': pharm_out}


# ===========================================================================
# Graph-level GRU readout
# ===========================================================================
class GraphGRU(nn.Module):
    """Cross-attention between atom and pharm sequences, then GRU pooling.

    1. Multi-head cross-attention (atom queries, pharm keys/values)
    2. Bidirectional GRU over attended sequence
    3. Mean pooling over time steps
    """

    def __init__(self, hid_dim, num_heads=4, bidirectional=True):
        super().__init__()
        self.attention = MultiHeadAttention(hid_dim, num_heads)
        self.gru = nn.GRU(
            hid_dim, hid_dim, batch_first=True,
            bidirectional=bidirectional,
        )
        self.direction = 2 if bidirectional else 1
        self.out_dim = self.direction * hid_dim

    def forward(self, atom_seq, pharm_seq):
        attn_out, _ = self.attention(atom_seq, pharm_seq, pharm_seq)
        gru_out, _ = self.gru(attn_out)
        return gru_out.mean(dim=1)


# ===========================================================================
# Multi-View Message Passing (stacked HeteroGNNLayers)
# ===========================================================================
class MVMP(nn.Module):
    """Stacks multiple HeteroGNNLayer for iterative message passing."""

    def __init__(self, hid_dim, depth=3, num_heads=4, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            HeteroGNNLayer(hid_dim, num_heads, dropout)
            for _ in range(depth)
        ])

    def forward(self, x_dict, edge_index_dict):
        for layer in self.layers:
            x_dict = layer(x_dict, edge_index_dict)
        return x_dict
