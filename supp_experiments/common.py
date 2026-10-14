"""
common.py — 补充实验公共模块（实验一：容量匹配消融 / 实验四：特征排列随机化）
==============================================================================

严格遵循《论文修改清单与补充实验方案》附录「补充实验代码框架」的模型定义、
参数量与超参数设定，仅做两处让实验可运行且结论稳健的适配：
  1. load_data: 真实数据（surfpro_train.csv / surfpro_test.csv，522 维特征），
     目标列顺序 [pCMC, gamma_CMC, Gamma_max, A_min, Pi_CMC, pC20]，Gamma_max 已乘 1e6；
  2. 训练协议按论文约定（per-target 建模、12.5% 验证集早停），
     因为文档实验矩阵即 6模型×6目标=36 组、2模型×6目标×(1基线+5排列)=72 组 = per-target。

- 掩码即每目标各自 dropna（与论文 per-target 管线一致）；
- Gamma_max 训练时用 1e6 缩放后的尺度，评估时还原为原始 1e-6 尺度输出 RMSE/MAE（R² 不变）。
"""
import os
import copy
import json
import itertools

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ---------------------------------------------------------------------------
# 0. 全局配置（来自文档）
# ---------------------------------------------------------------------------
SEED = 42
YSCALE_GMAX = 1e6
EPOCHS = 300          # 文档设定
PATIENCE = 30         # 文档设定
LR = 1e-3
BATCH_SIZE = 64
VAL_FRAC = 0.125      # 论文设定（与训练管线中 val_frac 一致）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 目标顺序（文档）：pCMC, gamma_CMC, Gamma_max, A_min, Pi_CMC, pC20
TARGET_NAMES = ["pCMC", "gamma_CMC", "Gamma_max", "A_min", "Pi_CMC", "pC20"]
TARGET_CSV_COLS = ["pCMC", "AW_ST_CMC", "Gamma_max", "Area_min", "Pi_CMC", "pC20"]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUPP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SUPP_DIR, "data")
RESULTS_DIR = os.path.join(SUPP_DIR, "results")
PREPARED_NPZ = os.path.join(DATA_DIR, "prepared_data.npz")


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# 1. 模型定义（与文档代码框架一致，输出改为单目标）
# ---------------------------------------------------------------------------
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class MLP(nn.Module):
    """全连接网络：num_layers-1 个隐藏块 + 输出层（文档 MLP-original=5层×488）"""
    def __init__(self, input_dim=522, num_layers=5, hidden_dim=488, dropout=0.049):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class RNN(nn.Module):
    """三层 LSTM，把 522 维特征看作 522 个时间步 × 1 维输入（文档结构）"""
    def __init__(self, input_dim=522, num_layers=3, hidden_size=64, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1, hidden_size=hidden_size, num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0, batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.unsqueeze(-1)              # (batch, 522, 1)
        _, (h_n, _) = self.lstm(x)
        h = h_n[-1]
        return self.head(h).squeeze(-1)


class TransformerModel(nn.Module):
    """Transformer 编码器（文档结构；注意：无位置编码 → 对特征列置换天然不变）"""
    def __init__(self, input_dim=522, d_model=64, nhead=4, num_layers=3,
                 dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)      # 每维特征投影到 d_model
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        x = x.unsqueeze(-1)              # (batch, 522, 1)
        x = self.input_proj(x)           # (batch, 522, d_model)
        x = self.encoder(x)
        x = x.mean(dim=1)                # 全局平均池化
        return self.head(x).squeeze(-1)


# ---------------------------------------------------------------------------
# 2. 参数量统计与容量搜索（来自文档 build_model_configs）
# ---------------------------------------------------------------------------
def _make_mlp_orig():
    return MLP(input_dim=522, num_layers=5, hidden_dim=488, dropout=0.049)


def _make_mlp_small():
    return MLP(input_dim=522, num_layers=3, hidden_dim=64, dropout=0.1)


def _make_rnn_orig():
    return RNN(input_dim=522, num_layers=3, hidden_size=64, dropout=0.1)


def _make_trans_orig():
    return TransformerModel(input_dim=522, d_model=64, nhead=4, num_layers=3,
                            dim_feedforward=256, dropout=0.1)


def search_rnn_large(n_mlp, tol=0.15):
    """搜索与 MLP-original 参数量对齐的 RNN 配置（在 ±15% 内取第一个）。"""
    cands = []
    for nl in [2, 3, 4, 5]:
        for hs in [128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 488, 512, 640, 768]:
            m = RNN(input_dim=522, num_layers=nl, hidden_size=hs, dropout=0.1)
            n = count_params(m)
            cands.append((abs(n - n_mlp) / n_mlp, n, nl, hs))
    cands.sort()
    _, n, nl, hs = cands[0]
    return {"num_layers": nl, "hidden_size": hs, "params": int(n),
            "ratio": round(n / n_mlp, 3), "best_fit": round(cands[0][0], 3), "tolerance": tol}


def search_trans_large(n_mlp, tol=0.15):
    """搜索与 MLP-original 参数量对齐的 Transformer 配置。"""
    cands = []
    for dm in [64, 96, 128, 160, 192, 224, 256, 320, 384, 512]:
        for nl in [2, 3, 4, 5]:
            for ff in [dm * 2, dm * 4]:
                m = TransformerModel(input_dim=522, d_model=dm, nhead=4, num_layers=nl,
                                     dim_feedforward=ff, dropout=0.1)
                n = count_params(m)
                cands.append((abs(n - n_mlp) / n_mlp, n, nl, dm, ff))
    cands.sort()
    _, n, nl, dm, ff = cands[0]
    return {"num_layers": nl, "d_model": dm, "dim_feedforward": ff, "params": int(n),
            "ratio": round(n / n_mlp, 3), "best_fit": round(cands[0][0], 3), "tolerance": tol}


def build_model_configs():
    """
    返回 6 个配置的工厂函数与本征参数量，容量搜索基于 MLP-original 的参数量。
    """
    mlp_orig = _make_mlp_orig()
    n_mlp = count_params(mlp_orig)
    rnn_orig = _make_rnn_orig()
    n_rnn = count_params(rnn_orig)
    trans_orig = _make_trans_orig()
    n_trans = count_params(trans_orig)
    mlp_small = _make_mlp_small()
    n_mlp_small = count_params(mlp_small)

    rnn_large_cfg = search_rnn_large(n_mlp)
    trans_large_cfg = search_trans_large(n_mlp)

    configs = {
        "MLP-original": {
            "params": int(n_mlp), "make": lambda: _make_mlp_orig(),
            "desc": "5层×488隐藏维 (论文/文档原型)",
            "arch": {"model": "MLP", "num_layers": 5, "hidden_dim": 488, "dropout": 0.049},
        },
        "MLP-small": {
            "params": int(n_mlp_small), "make": lambda: _make_mlp_small(),
            "desc": "3层×64隐藏维 (小容量，非序列化反向对照)",
            "arch": {"model": "MLP", "num_layers": 3, "hidden_dim": 64, "dropout": 0.1},
        },
        "RNN-original": {
            "params": int(n_rnn), "make": lambda: _make_rnn_orig(),
            "desc": "3层×64隐藏单元 (论文/文档原型)",
            "arch": {"model": "RNN", "num_layers": 3, "hidden_size": 64, "dropout": 0.1},
        },
        "RNN-large": {
            "params": int(rnn_large_cfg["params"]), "make": lambda: RNN(
                input_dim=522, num_layers=rnn_large_cfg["num_layers"],
                hidden_size=rnn_large_cfg["hidden_size"], dropout=0.1),
            "desc": f"容量匹配 (与 MLP-original 对齐: {rnn_large_cfg['num_layers']}层×"
                    f"{rnn_large_cfg['hidden_size']}hs, ratio={rnn_large_cfg['ratio']}x)",
            "arch": rnn_large_cfg,
        },
        "Transformer-original": {
            "params": int(n_trans), "make": lambda: _make_trans_orig(),
            "desc": "3层×d_model=64 (论文/文档原型)",
            "arch": {"model": "Transformer", "d_model": 64, "nhead": 4, "num_layers": 3,
                     "dim_feedforward": 256, "dropout": 0.1},
        },
        "Transformer-large": {
            "params": int(trans_large_cfg["params"]), "make": lambda: TransformerModel(
                input_dim=522, d_model=trans_large_cfg["d_model"], nhead=4,
                num_layers=trans_large_cfg["num_layers"],
                dim_feedforward=trans_large_cfg["dim_feedforward"], dropout=0.1),
            "desc": f"容量匹配 (与 MLP-original 对齐: {trans_large_cfg['num_layers']}层×"
                    f"dm{trans_large_cfg['d_model']}×ff{trans_large_cfg['dim_feedforward']}, "
                    f"ratio={trans_large_cfg['ratio']}x)",
            "arch": trans_large_cfg,
        },
    }
    # 附加说明
    configs["_capacity_report"] = {
        "MLP-original_params": int(n_mlp),
        "RNN-original_params": int(n_rnn),
        "Transformer-original_params": int(n_trans),
        "RNN-large": rnn_large_cfg,
        "Transformer-large": trans_large_cfg,
        "MLP/RNN-original_ratio": round(n_mlp / n_rnn, 2),
        "MLP/Transformer-original_ratio": round(n_mlp / n_trans, 2),
    }
    return configs


# ---------------------------------------------------------------------------
# 3. 数据加载与 per-target 切分
# ---------------------------------------------------------------------------
def load_prepared():
    d = np.load(PREPARED_NPZ)
    return {
        "X_train": d["X_train"], "X_test": d["X_test"],
        "Y_train": d["Y_train"], "Y_test": d["Y_test"],
        "y_train_mask": d["y_train_mask"], "y_test_mask": d["y_test_mask"],
    }


def get_target_data(data, j):
    """返回第 j 个目标的有效训练/测试样本（per-target dropna，Gamma_max 已含 1e6 缩放）。"""
    X_train = data["X_train"][data["y_train_mask"][:, j]]
    y_train = data["Y_train"][data["y_train_mask"][:, j], j]
    X_test = data["X_test"][data["y_test_mask"][:, j]]
    y_test = data["Y_test"][data["y_test_mask"][:, j], j]
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# 4. 训练与评估（per-target；12.5% 验证集早停）
# ---------------------------------------------------------------------------
def train_one_model(model, X_train, y_train, X_val, y_val, verbose=False):
    """在训练集上训练，验证集早停（patience），返回 (恢复到最优验证损失状态的模型, 最优epoch)。"""
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(PATIENCE // 3, 5))

    tr_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                      torch.tensor(y_train, dtype=torch.float32)),
        batch_size=BATCH_SIZE, shuffle=True)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    best_epoch = 1
    wait = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for X_b, y_b in tr_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()
        scheduler.step(val_loss)
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    if verbose:
        print(f"    early stop @ epoch {epoch}, best_epoch={best_epoch}, best_val_loss={best_val:.6f}")
    return model, best_epoch


def evaluate(model, X_test, y_test, unscale=None):
    """在测试集上计算 R²/RMSE/MAE；unscale 用于 Gamma_max（预测与真值一起 ×1e6 → 还原为原始 1e-6 尺度）。

    注意：y 在 prepare_data.py 里已按 YSCALE_GMAX=1e6 缩放（模型在该尺度上训练），
    因此评估时若还原，必须把 y_pred 与 y_true 一并除以 unscale，
    否则只有预测被缩小 1e6 倍会得到「预测≈0」的假 R²（2026-10-10 修复）。
    """
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        y_pred = model(X_t).cpu().numpy().astype(float)
    y_true = np.asarray(y_test, dtype=float)
    if unscale is not None:
        y_pred = y_pred / unscale
        y_true = y_true / unscale
    mask = ~np.isnan(y_true)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) < 5:
        return {"R2": np.nan, "RMSE": np.nan, "MAE": np.nan}
    return {
        "R2": round(float(r2_score(y_true, y_pred)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 4),
    }