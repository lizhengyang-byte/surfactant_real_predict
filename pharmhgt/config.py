from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class PharmHGTConfig:
    data_dir: str = './data'
    train_csv: str = 'surfpro_imputed.csv'
    test_csv: str = 'surfpro_test.csv'
    target_col: str = 'pCMC'
    smiles_col: str = 'SMILES'
    graph_cache_dir: str = 'data/features/pharmhgt_graph'

    atom_dim: int = 42
    bond_dim: int = 14
    pharm_dim: int = 194
    reac_dim: int = 34
    hid_dim: int = 256
    num_heads: int = 4
    depth: int = 3
    dropout: float = 0.1
    num_task: int = 1

    batch_size: int = 32
    epochs: int = 300
    lr: float = 5e-4
    weight_decay: float = 1e-5
    warmup_epochs: int = 5
    init_lr: float = 1e-5
    max_lr: float = 5e-4
    final_lr: float = 1e-6
    num_fold: int = 5
    seed: int = 42
    device: Optional[str] = None

    save_dir: str = 'models/pharmhgt'
    gradient_clip: float = 5.0
    patience: int = 40

    def __post_init__(self):
        if self.device is None:
            import torch
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
