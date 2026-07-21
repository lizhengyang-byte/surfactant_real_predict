"""
Heterogeneous molecular graph construction for PharmHGT.

Builds a PyG HeteroData graph with:
  - Node types: 'atom' (atoms), 'pharm' (BRICS pharmacophore fragments)
  - Edge types: ('atom','bond','atom'), ('pharm','react','pharm'), ('atom','junc','pharm')

Features follow the PharmHGT paper design (42/14/194/34-dim).
"""

import hashlib, json, os, warnings, zlib
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData, Batch
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
import rdkit.rdBase as rkrb
rkrb.DisableLog('rdApp.error')
from rdkit.Chem import (
    rdMolDescriptors, Descriptors, AllChem, MACCSkeys, BRICS, ChemicalFeatures
)
from rdkit.Chem.rdchem import BondType as BT, HybridizationType
from rdkit import RDConfig

import os
os.environ['RDKIT_SANITIZE'] = 'false'
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Pharmacophore feature factory (RDKit BaseFeatures)
# ---------------------------------------------------------------------------
_fdef_path = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
_pharm_factory = ChemicalFeatures.BuildFeatureFactory(_fdef_path)

# 27 pharmacophore property types from RDKit BaseFeatures
_PHARM_TYPES = [
    i.split('.')[1] for i in _pharm_factory.GetFeatureDefs().keys()
]

# ---------------------------------------------------------------------------
# Atom feature helpers (42-dim)
# ---------------------------------------------------------------------------
_ATOM_ELEMENTS = [6, 7, 8, 9, 15, 16, 17, 35, 53]
_ATOM_FEATURE_SCHEMA = {
    'atomic_num': _ATOM_ELEMENTS,
    'degree': list(range(6)),
    'formal_charge': [-1, -2, 1, 2, 0],
    'chiral_tag': [0, 1, 2, 3],
    'num_hs': list(range(5)),
    'hybridization': [
        HybridizationType.SP, HybridizationType.SP2, HybridizationType.SP3,
        HybridizationType.SP3D, HybridizationType.SP3D2,
    ],
}


def _onehot(value, choices):
    out = [0] * (len(choices) + 1)
    idx = choices.index(value) if value in choices else -1
    out[idx] = 1
    return out


def atom_features_42(atom: Chem.Atom) -> list:
    atomic_num = atom.GetAtomicNum()
    feats = (
        _onehot(atomic_num, _ATOM_ELEMENTS)
        + _onehot(atom.GetTotalDegree(), _ATOM_FEATURE_SCHEMA['degree'])
        + _onehot(atom.GetFormalCharge(), _ATOM_FEATURE_SCHEMA['formal_charge'])
        + _onehot(int(atom.GetChiralTag()), _ATOM_FEATURE_SCHEMA['chiral_tag'])
        + _onehot(atom.GetTotalNumHs(), _ATOM_FEATURE_SCHEMA['num_hs'])
        + _onehot(int(atom.GetHybridization()), _ATOM_FEATURE_SCHEMA['hybridization'])
        + [1.0 if atom.GetIsAromatic() else 0.0]
        + [atom.GetMass() * 0.01]
    )
    assert len(feats) == 42, f"atom feature length mismatch: {len(feats)}"
    return feats


# ---------------------------------------------------------------------------
# Bond feature helpers (14-dim)
# ---------------------------------------------------------------------------
_BOND_STEREO_TYPES = list(range(6))


def bond_features_14(bond: Chem.Bond) -> list:
    if bond is None:
        return [1.0] + [0.0] * 13
    bt = bond.GetBondType()
    feats = (
        [0.0]
        + [
            1.0 if bt == BT.SINGLE else 0.0,
            1.0 if bt == BT.DOUBLE else 0.0,
            1.0 if bt == BT.TRIPLE else 0.0,
            1.0 if bt == BT.AROMATIC else 0.0,
        ]
        + [1.0 if bond.GetIsConjugated() else 0.0]
        + [1.0 if bond.IsInRing() else 0.0]
        + _onehot(int(bond.GetStereo()), _BOND_STEREO_TYPES)
    )
    assert len(feats) == 14, f"bond feature length mismatch: {len(feats)}"
    return feats


# ---------------------------------------------------------------------------
# Pharmacophore fragment features (194-dim)
# ---------------------------------------------------------------------------
def _pharm_property_types(mol) -> list:
    types = [i.GetType() for i in _pharm_factory.GetFeaturesForMol(mol)]
    return [1.0 if t in types else 0.0 for t in _PHARM_TYPES]


def _pharm_fragment_features(mol: Chem.Mol) -> list:
    try:
        maccs = list(MACCSkeys.GenMACCSKeys(mol))
    except Exception:
        maccs = [0] * 167
    ptypes = _pharm_property_types(mol)
    feats = maccs + ptypes
    if len(feats) != 194:
        return [0.0] * 194
    return feats


# ---------------------------------------------------------------------------
# BRICS reaction features (34-dim)
# ---------------------------------------------------------------------------
def _brics_bond_feature(action):
    start = int(action[0]) if action[0] not in ('7a', '7b') else 7
    end = int(action[1]) if action[1] not in ('7a', '7b') else 7
    emb_start = [0.0] * 17
    emb_end = [0.0] * 17
    emb_start[start] = 1.0
    emb_end[end] = 1.0
    return emb_start + emb_end


# ---------------------------------------------------------------------------
# Fragment decomposition via BRICS
# ---------------------------------------------------------------------------
def _decompose_fragments(mol: Chem.Mol):
    """Decompose molecule into BRICS fragments.

    Returns:
        atom_to_pharm: dict {atom_idx: pharm_idx}
        pharm_feats: dict {pharm_idx: 194-dim feature list}
    """
    from rdkit.Chem.BRICS import FindBRICSBonds

    break_bonds = [
        mol.GetBondBetweenAtoms(i[0][0], i[0][1]).GetIdx()
        for i in FindBRICSBonds(mol)
    ]
    if not break_bonds:
        tmp = mol
    else:
        tmp = Chem.FragmentOnBonds(mol, break_bonds, addDummies=False)

    frags_idx_lst = Chem.GetMolFrags(tmp)
    atom_to_pharm = {}
    pharm_feats = {}
    for pharm_idx, frag_idx in enumerate(frags_idx_lst):
        for atom_idx in frag_idx:
            atom_to_pharm[atom_idx] = pharm_idx
        frag_mol = Chem.MolFromSmiles(
            Chem.MolFragmentToSmiles(mol, frag_idx)
        )
        if frag_mol is not None:
            pharm_feats[pharm_idx] = _pharm_fragment_features(frag_mol)
        else:
            pharm_feats[pharm_idx] = [0.0] * 194
    return atom_to_pharm, pharm_feats


def _get_brics_reaction_edges(mol, atom_to_pharm):
    """Extract BRICS reaction edges between fragments.

    Returns:
        edges: list of [(pharm_i, pharm_j)]
        feats: list of [34-dim feature]
    """
    from rdkit.Chem.BRICS import FindBRICSBonds

    brics_bonds = FindBRICSBonds(mol)
    edges = []
    feats = []
    for item in brics_bonds:
        begin, end = item[0]
        rule = item[1]
        p_begin = atom_to_pharm.get(begin)
        p_end = atom_to_pharm.get(end)
        if p_begin is not None and p_end is not None:
            edges.append((p_begin, p_end))
            feat = _brics_bond_feature(rule)
            feats.append(feat)
    return edges, feats


# ---------------------------------------------------------------------------
# Main conversion: SMILES -> HeteroData
# ---------------------------------------------------------------------------
def sanitize_safe(mol):
    try:
        mol.UpdatePropertyCache()
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE)
        return mol
    except Exception:
        try:
            mol.UpdatePropertyCache()
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_FINDRADICALS | Chem.SANITIZE_SETAROMATICITY)
            return mol
        except Exception:
            return None


def smiles_to_heterograph(smiles: str) -> HeteroData:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = sanitize_safe(mol)
    if mol is None:
        return None

    try:
        AllChem.ComputeGasteigerCharges(mol)
    except Exception:
        pass

    # --- Fragment decomposition ---
    atom_to_pharm, pharm_feats_dict = _decompose_fragments(mol)
    brics_edges, brics_feats = _get_brics_reaction_edges(mol, atom_to_pharm)

    # --- Atom features ---
    num_atoms = mol.GetNumAtoms()
    atom_feats = [atom_features_42(mol.GetAtomWithIdx(i)) for i in range(num_atoms)]
    atom_x = torch.tensor(atom_feats, dtype=torch.float)

    # --- Pharm (fragment) features ---
    num_pharms = len(pharm_feats_dict)
    if num_pharms > 0:
        pharm_x = torch.tensor(
            [pharm_feats_dict[i] for i in sorted(pharm_feats_dict.keys())],
            dtype=torch.float
        )
    else:
        pharm_x = torch.zeros((0, 194), dtype=torch.float)

    # --- Bond edges (+ reverse) ---
    bond_src, bond_dst, bond_attr = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = bond_features_14(bond)
        bond_src.extend([i, j])
        bond_dst.extend([j, i])
        bond_attr.extend([feat, feat])
    bond_edge_index = torch.tensor([bond_src, bond_dst], dtype=torch.long) if bond_src else torch.zeros((2, 0), dtype=torch.long)
    bond_edge_attr = torch.tensor(bond_attr, dtype=torch.float) if bond_attr else torch.zeros((0, 14), dtype=torch.float)

    # --- Reaction edges (+ reverse) ---
    reac_src, reac_dst, reac_attr = [], [], []
    for (p_i, p_j), feat in zip(brics_edges, brics_feats):
        reac_src.extend([p_i, p_j])
        reac_dst.extend([p_j, p_i])
        reac_attr.extend([feat, feat])
    reac_edge_index = torch.tensor([reac_src, reac_dst], dtype=torch.long) if reac_src else torch.zeros((2, 0), dtype=torch.long)
    reac_edge_attr = torch.tensor(reac_attr, dtype=torch.float) if reac_attr else torch.zeros((0, 34), dtype=torch.float)

    # --- Junction edges (atom -> pharm and pharm -> atom) ---
    junc_ap_src, junc_ap_dst = [], []
    for atom_idx, pharm_idx in atom_to_pharm.items():
        junc_ap_src.append(atom_idx)
        junc_ap_dst.append(pharm_idx)
    junc_ap = torch.tensor([junc_ap_src, junc_ap_dst], dtype=torch.long) if junc_ap_src else torch.zeros((2, 0), dtype=torch.long)
    junc_pa = torch.tensor([junc_ap_dst, junc_ap_src], dtype=torch.long) if junc_ap_src else torch.zeros((2, 0), dtype=torch.long)

    # --- Build HeteroData ---
    data = HeteroData()
    data['atom'].x = atom_x
    data['pharm'].x = pharm_x
    data['atom', 'bond', 'atom'].edge_index = bond_edge_index
    data['atom', 'bond', 'atom'].edge_attr = bond_edge_attr
    data['pharm', 'react', 'pharm'].edge_index = reac_edge_index
    data['pharm', 'react', 'pharm'].edge_attr = reac_edge_attr
    data['atom', 'junc', 'pharm'].edge_index = junc_ap
    data['pharm', 'junc', 'atom'].edge_index = junc_pa

    return data


# ---------------------------------------------------------------------------
# Dataset for training
# ---------------------------------------------------------------------------
class SurfactantGraphDataset(Dataset):
    """Dataset that converts SMILES to HeteroData graphs on-the-fly.

    Invalid SMILES are pre-filtered during construction.
    """

    def __init__(self, df, smiles_col='SMILES', target_col='pCMC', cache_dir=None, verbose=True):
        self.smiles_list = []
        self.targets = []
        self.cache_dir = cache_dir

        invalid = 0
        for _, row in df.iterrows():
            smi = row[smiles_col]
            g = smiles_to_heterograph(smi)
            if g is not None:
                self.smiles_list.append(smi)
                self.targets.append(row[target_col])
            else:
                invalid += 1

        self.targets = np.array(self.targets, dtype=np.float32)
        self._graphs = [None] * len(self.smiles_list)

        if verbose and invalid > 0:
            print(f'    (skipped {invalid} invalid SMILES)')

        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        if self._graphs[idx] is not None:
            g = self._graphs[idx]
        else:
            g = smiles_to_heterograph(self.smiles_list[idx])
            if g is None:
                g = HeteroData()
                g['atom'].x = torch.zeros((1, 42), dtype=torch.float)
                g['pharm'].x = torch.zeros((1, 194), dtype=torch.float)
            self._graphs[idx] = g
        return g, torch.tensor(self.targets[idx], dtype=torch.float)


def collate_graphs(batch):
    graphs = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    batch_g = Batch.from_data_list(graphs)
    return batch_g, labels


# ---------------------------------------------------------------------------
# High-level API (matches existing project pattern)
# ---------------------------------------------------------------------------
def load_and_build_graphs(
    train_csv='./data/surfpro_imputed.csv',
    test_csv='./data/surfpro_test.csv',
    target_col='pCMC',
    smiles_col='SMILES',
    cache_dir=None,
    force_recompute=False,
    verbose=True,
):
    """Load CSV data, build graphs, and return dataloaders."""
    if cache_dir is None:
        cache_dir = os.path.join('data', 'features', 'pharmhgt_graph')

    df_train = pd.read_csv(train_csv).dropna(subset=[target_col])
    df_test = pd.read_csv(test_csv).dropna(subset=[target_col])

    if verbose:
        print(f"Train: {len(df_train)} molecules, Test: {len(df_test)} molecules")

    train_dataset = SurfactantGraphDataset(
        df_train, smiles_col, target_col, cache_dir, verbose
    )
    test_dataset = SurfactantGraphDataset(
        df_test, smiles_col, target_col, cache_dir, verbose
    )

    return train_dataset, test_dataset
