"""
PharmHGT — Pharmacophoric-constrained Heterogeneous Graph Transformer.

Independent implementation based on:
  Jiang et al., "Pharmacophoric-constrained heterogeneous graph transformer
  model for molecular property prediction", Commun Chem 6, 60 (2023).

Architecture:
  1. Feature projection (atom/bond/pharm/reac/junc -> hid_dim)
  2. Multi-View Message Passing (MVMP) x depth layers
  3. Graph-level GRU readout with cross-attention
  4. MLP prediction head

Uses PyTorch Geometric (HeteroData) instead of DGL.
"""

import torch
import torch.nn as nn
from torch_geometric.utils import scatter
from pharmhgt.layers import MVMP, GraphGRU


class PharmHGT(nn.Module):
    def __init__(
        self,
        atom_dim=42,
        bond_dim=14,
        pharm_dim=194,
        reac_dim=34,
        hid_dim=256,
        depth=3,
        num_heads=4,
        dropout=0.1,
        num_task=1,
    ):
        super().__init__()

        self.hid_dim = hid_dim

        # --- Feature projection ---
        self.atom_proj = nn.Linear(atom_dim, hid_dim)
        self.pharm_proj = nn.Linear(pharm_dim, hid_dim)
        self.junc_proj = nn.Linear(atom_dim + pharm_dim, hid_dim)
        self.activation = nn.ReLU()

        # --- Multi-View Message Passing ---
        self.mp = MVMP(hid_dim, depth, num_heads, dropout)

        # --- Graph readout ---
        self.readout = GraphGRU(hid_dim, num_heads, bidirectional=True)

        # --- Prediction head ---
        readout_dim = self.readout.out_dim
        self.predictor = nn.Sequential(
            nn.Linear(readout_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, num_task),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _project_features(self, data):
        atom_feat = data['atom'].x
        pharm_feat = data['pharm'].x

        atom_x = self.activation(self.atom_proj(atom_feat))
        pharm_x = self.activation(self.pharm_proj(pharm_feat))

        n_atom = atom_feat.size(0)
        n_pharm = pharm_feat.size(0)
        device = atom_feat.device

        atom_junc = torch.cat([
            atom_feat,
            torch.zeros(n_atom, 194, device=device),
        ], dim=-1)
        pharm_junc = torch.cat([
            torch.zeros(n_pharm, 42, device=device),
            pharm_feat,
        ], dim=-1)

        atom_junc = self.activation(self.junc_proj(atom_junc))
        pharm_junc = self.activation(self.junc_proj(pharm_junc))

        return {'atom': atom_x, 'pharm': pharm_x}, \
               {'atom': atom_junc, 'pharm': pharm_junc}

    def _padded_sequence(self, x, batch):
        batch_size = int(batch.max().item()) + 1
        device = x.device

        num_per = scatter(
            torch.ones(x.size(0), device=device),
            batch, dim=0, reduce='sum'
        ).long()
        max_len = int(num_per.max().item())

        padded = torch.zeros(batch_size, max_len, self.hid_dim, device=device)
        for i in range(batch_size):
            mask = batch == i
            seq = x[mask]
            padded[i, :seq.size(0)] = seq
        return padded

    def forward(self, data):
        edge_index_dict = data.edge_index_dict
        x_dict, _ = self._project_features(data)

        # --- Message passing ---
        x_dict = self.mp(x_dict, edge_index_dict)

        atom_x = x_dict['atom']
        pharm_x = x_dict['pharm']

        # --- Batch separation ---
        atom_batch = data['atom'].batch
        pharm_batch = data['pharm'].batch if 'pharm' in data.node_types and hasattr(data['pharm'], 'batch') else \
                      torch.zeros(pharm_x.size(0), dtype=torch.long, device=atom_x.device)

        # --- Padding to sequences for GRU ---
        atom_seq = self._padded_sequence(atom_x, atom_batch)
        pharm_seq = self._padded_sequence(pharm_x, pharm_batch)

        # --- Readout ---
        graph_embed = self.readout(atom_seq, pharm_seq)

        # --- Prediction ---
        out = self.predictor(graph_embed).squeeze(-1)
        return out
