"""
SurfPredict — Unified Model Prediction API
===========================================

从 runs/ 自动加载已训练的模型，提供统一的分子性质预测接口，
支持多目标（pCMC / AW_ST_CMC / Gamma_max / Area_min / Pi_CMC / pC20）。

用法:
    from use_models import SmilesPredict, SmilesPredictor, list_models

    # ---- 最简单的调用形式 ----
    pred = SmilesPredict('CCO', target='pCMC')                    # pCMC（缺省）
    pred = SmilesPredict('CCO', model_name='catboost', target='AW_ST_CMC')
    pred = SmilesPredict(['CCO', 'CCC(=O)O'], target='Gamma_max') # 批量
    pred, feats = SmilesPredict('CCO', target='pCMC', return_features=True)

    # ---- 查看某 target 的可用模型 ----
    df = list_models(target='AW_ST_CMC')
"""

import csv
import json
import os
import re
import warnings
import sys
from typing import Optional, Union, List

import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings('ignore')

# ============================================================================
# Path setup
# ============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# smiles_to_features_pharmhgt.py 位于 train/train_pCMC_models/ 下
_TRAIN_PCMC_PATH = os.path.join(PROJECT_ROOT, 'train', 'train_pCMC_models')
if _TRAIN_PCMC_PATH not in sys.path:
    sys.path.insert(0, _TRAIN_PCMC_PATH)

from smiles_to_features_pharmhgt import smiles_to_features_pharmhgt as _featurize_single

RUNS_DIR = os.path.join(PROJECT_ROOT, 'runs')
INDEX_PATH = os.path.join(RUNS_DIR, '_runs_index.csv')

# PyTorch availability check
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ============================================================================
# PyTorch Model Definitions (must match training scripts exactly)
# ============================================================================

if TORCH_AVAILABLE:

    class _MLPRegressor(nn.Module):
        """MLP regression network (matches train_mlp_use_pharmhgt_features.py)."""

        def __init__(self, input_dim, hidden_dim, n_layers, dropout, activation='relu'):
            super().__init__()
            layers = []
            prev_dim = input_dim
            for _ in range(n_layers):
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.BatchNorm1d(hidden_dim))
                act = {'relu': nn.ReLU(), 'gelu': nn.GELU(), 'leaky_relu': nn.LeakyReLU(0.1)}
                layers.append(act.get(activation, nn.ReLU()))
                layers.append(nn.Dropout(dropout))
                prev_dim = hidden_dim
            layers.append(nn.Linear(prev_dim, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x).squeeze(-1)

    class _RNNRegressor(nn.Module):
        """RNN-LSTM regression network (matches train_rnn_use_pharmhgt_features.py)."""

        def __init__(self, _input_dim, hidden_dim, n_layers, dropout, _activation='relu'):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=1,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                batch_first=True,
                dropout=dropout if n_layers > 1 else 0,
            )
            self.fc = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            x = x.unsqueeze(-1)
            lstm_out, _ = self.lstm(x)
            last_out = lstm_out[:, -1, :]
            return self.fc(last_out).squeeze(-1)

    class _PositionalEncoding(nn.Module):
        """Sinusoidal positional encoding for Transformer."""

        def __init__(self, d_model, max_len=1024):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe.unsqueeze(0))

        def forward(self, x):
            return x + self.pe[:, :x.size(1), :]

    class _TransformerRegressor(nn.Module):
        """Transformer Encoder regression network."""

        def __init__(self, input_dim, d_model=128, nhead=4, num_layers=3,
                     dim_feedforward=256, dropout=0.1, activation='relu'):
            super().__init__()
            self.input_proj = nn.Linear(1, d_model)
            self.pos_encoder = _PositionalEncoding(d_model, max_len=input_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model, nhead, dim_feedforward, dropout,
                activation=activation, batch_first=True,
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
            self.fc = nn.Linear(d_model, 1)

        def forward(self, x):
            x = x.unsqueeze(-1)
            x = self.input_proj(x)
            x = self.pos_encoder(x)
            x = self.transformer_encoder(x)
            x = x.mean(dim=1)
            return self.fc(x).squeeze(-1)


# ============================================================================
# Internal helpers
# ============================================================================

#: 匹配运行目录名: catboost_20260722_181638 或 pCMC_catboost_20260722_181638
_RUN_DIR_PATTERN = re.compile(r'^(.+)_(\d{8}_\d{6})$')

TARGETS = {'pCMC', 'AW_ST_CMC', 'Gamma_max', 'Area_min', 'Pi_CMC', 'pC20'}


def _parse_run_dir_name(dirname: str, target: str):
    """从目录名中提取模型名和时间戳。

    兼容新旧两种命名:
      catboost_20260722_181638       → model=catboost
      pCMC_catboost_20260722_181638  → model=catboost

    Returns:
        (model_name, timestamp_str) 或 (None, None)
    """
    m = _RUN_DIR_PATTERN.match(dirname)
    if not m:
        return None, None
    run_name = m.group(1)
    ts = m.group(2)
    # 去掉 target 前缀（如果存在）
    prefix = target + '_'
    if run_name.startswith(prefix):
        model_name = run_name[len(prefix):]
    else:
        model_name = run_name
    return model_name, ts


def _discover_model_runs(target: str = 'pCMC'):
    """扫描 runs/{target}/ 下可用的模型运行。

    Returns:
        dict: {model_name: [(timestamp_str, run_dir_path), ...]},
              按时间戳降序排列。跳过没有 model.pkl 的目录。
    """
    target_dir = os.path.join(RUNS_DIR, target)
    if not os.path.isdir(target_dir):
        return {}

    models = {}
    for entry in os.listdir(target_dir):
        dirpath = os.path.join(target_dir, entry)
        if not os.path.isdir(dirpath):
            continue
        if not os.path.isfile(os.path.join(dirpath, 'model.pkl')):
            continue
        model_name, ts = _parse_run_dir_name(entry, target)
        if model_name is None:
            continue
        models.setdefault(model_name, []).append((ts, dirpath))

    for name in models:
        models[name].sort(key=lambda x: x[0], reverse=True)
    return models


def _load_config(run_dir: str) -> dict:
    """加载运行目录中的 config.json。"""
    config_path = os.path.join(run_dir, 'config.json')
    if os.path.isfile(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def _get_y_scale(run_dir: str) -> float:
    """从 config.json 中读取 y_scale 缩放因子。

    Gamma_max 使用 1_000_000，其他 target 为 1（即不缩放）。
    """
    config = _load_config(run_dir)
    return float(config.get('y_scale', 1))


def _load_sklearn_model(run_dir):
    """Load an sklearn-compatible model via joblib."""
    return joblib.load(os.path.join(run_dir, 'model.pkl'))


def _load_torch_model(run_dir, device='cpu'):
    """Rebuild a PyTorch model from saved checkpoint metadata + state_dict."""
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required to load deep learning models. "
            "Install with: pip install torch"
        )
    ckpt = torch.load(
        os.path.join(run_dir, 'model.pkl'),
        map_location=device,
        weights_only=True,
    )
    model_type = ckpt['model_type']
    input_dim = ckpt.get('input_dim', 522)

    builders = {
        'mlp': lambda: _MLPRegressor(
            input_dim, ckpt['hidden_dim'], ckpt['n_layers'],
            ckpt['dropout'], ckpt.get('activation', 'relu'),
        ),
        'rnn': lambda: _RNNRegressor(
            input_dim, ckpt['hidden_dim'], ckpt['n_layers'],
            ckpt['dropout'], ckpt.get('activation', 'relu'),
        ),
        'transformer': lambda: _TransformerRegressor(
            input_dim,
            d_model=ckpt.get('d_model', 128),
            nhead=ckpt.get('nhead', 4),
            num_layers=ckpt.get('num_layers', 3),
            dim_feedforward=ckpt.get('dim_feedforward', 256),
            dropout=ckpt.get('dropout', 0.1),
            activation=ckpt.get('activation', 'relu'),
        ),
    }
    if model_type not in builders:
        raise ValueError(f"Unknown model type: {model_type}")

    model = builders[model_type]()
    model.load_state_dict(ckpt['state_dict'])
    model.to(device)
    model.eval()
    return model


def _load_single_model(run_dir, device='cpu'):
    """Load a single model (auto-detect PyTorch vs sklearn)."""
    model_path = os.path.join(run_dir, 'model.pkl')
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if TORCH_AVAILABLE:
        try:
            ckpt = torch.load(model_path, map_location='cpu', weights_only=True)
            if 'model_type' in ckpt and 'state_dict' in ckpt:
                return _load_torch_model(run_dir, device)
        except Exception:
            pass

    return _load_sklearn_model(run_dir)


def _is_torch_model(model) -> bool:
    """Check if a loaded model is a PyTorch nn.Module."""
    return TORCH_AVAILABLE and isinstance(model, nn.Module)


# ============================================================================
# Public API
# ============================================================================

class SmilesPredictor:
    """Unified molecular property prediction interface.

    支持多目标预测，自动管理模型加载、特征计算和预测。

    Examples:
        >>> predictor = SmilesPredictor(target='pCMC')
        >>> predictor.predict('CCO')

        >>> predictor = SmilesPredictor(target='AW_ST_CMC', model_name='catboost')
        >>> predictor.predict(['CCO', 'CCC(=O)O'])

        >>> predictor = SmilesPredictor(target='Gamma_max')
        >>> predictor.predict('CCO')  # 自动还原 Y_SCALE，返回 mol/m²
    """

    def __init__(self, model_name: str = 'best',
                 target: str = 'pCMC',
                 run_dir: Optional[str] = None,
                 device: str = 'cpu'):
        """
        Args:
            model_name: 'best'（自动选最优）, 模型名如 'catboost'/'mlp', 或 'all'（集成）。
            target: 预测目标，可选 'pCMC', 'AW_ST_CMC', 'Gamma_max',
                    'Area_min', 'Pi_CMC', 'pC20'。默认 'pCMC'。
            run_dir: 指定运行目录路径（优先级高于 model_name）。
            device: Torch 设备（仅深度学习模型使用）。
        """
        self.device = device
        self.model_name = model_name
        self.target = target
        self.run_dir = run_dir
        self.y_scale = 1.0  # 从 config.json 读取，默认不缩放
        self._model = None
        self._models = {}

        if run_dir:
            self.load_from_dir(run_dir)
        elif model_name == 'best':
            self.load_best()
        elif model_name == 'all':
            self.load_all()
        else:
            self.load(model_name)

    # ---- Model discovery ---------------------------------------------------

    def available_models(self) -> pd.DataFrame:
        """返回当前 target 下所有训练模型的性能指标。"""
        return list_models(target=self.target)

    # ---- Loading -----------------------------------------------------------

    def load_best(self):
        """自动加载当前 target 下 test_rmse 最低的模型。"""
        df = self.available_models()
        if df.empty:
            raise RuntimeError(
                f"No trained models found for target '{self.target}'. "
                f"Run a training script first."
            )
        best = df.iloc[0]
        name = best['model']
        dir_rel = best['run_dir']
        # 解析实际路径（兼容新旧格式）
        dir_abs = _resolve_run_dir(dir_rel, self.target)
        if dir_abs is None:
            raise FileNotFoundError(
                f"Model '{name}' for target '{self.target}' listed in index "
                f"but no model.pkl found on disk."
            )
        rmse = best.get('test_rmse', 'N/A')

        print(f"[SurfPredict] Best model ({self.target}): {name}  (test_rmse={rmse})")
        self.model_name = name
        self.run_dir = dir_abs
        self._load_from_dir(dir_abs)

    def load(self, model_name: str):
        """加载当前 target 下指定模型的最新运行。

        Args:
            model_name: 如 'catboost', 'xgboost', 'mlp' 等。
        """
        models = _discover_model_runs(self.target)
        if model_name not in models:
            raise KeyError(
                f"Model '{model_name}' not found for target '{self.target}'. "
                f"Available: {sorted(models)}\n"
                f"Use list_models(target='{self.target}') for full details."
            )
        ts, dirpath = models[model_name][0]
        print(f"[SurfPredict] Loading {model_name} ({self.target}, latest: {ts})")
        self.model_name = model_name
        self.run_dir = dirpath
        self._load_from_dir(dirpath)

    def load_from_dir(self, run_dir: str):
        """从指定运行目录加载模型。"""
        self.run_dir = os.path.abspath(run_dir)
        self._load_from_dir(self.run_dir)
        # 从目录名推断 target 和 model_name
        parent = os.path.basename(os.path.dirname(self.run_dir))
        if parent in TARGETS:
            self.target = parent
        dirname = os.path.basename(self.run_dir)
        for t in TARGETS:
            model_name, _ = _parse_run_dir_name(dirname, t)
            if model_name:
                self.model_name = model_name
                break
        if not self.model_name:
            m = _RUN_DIR_PATTERN.match(dirname)
            if m:
                self.model_name = m.group(1)

    def load_all(self):
        """加载当前 target 下所有可用模型做集成预测。"""
        models = _discover_model_runs(self.target)
        if not models:
            raise RuntimeError(f"No trained models found for target '{self.target}'.")

        loaded = 0
        for name in models:
            _, dirpath = models[name][0]
            try:
                self._models[name] = _load_single_model(dirpath, self.device)
                loaded += 1
            except Exception as e:
                print(f"  [SurfPredict] Failed to load {name}: {e}")

        self.model_name = 'all'
        print(f"[SurfPredict] Ensemble ({self.target}): {loaded}/{len(models)} models loaded")

    def _load_from_dir(self, run_dir):
        """(internal) 加载模型 + 读取 y_scale。"""
        self._model = _load_single_model(run_dir, self.device)
        self.y_scale = _get_y_scale(run_dir)

    # ---- Prediction --------------------------------------------------------

    def predict(self, smiles, return_features: bool = False):
        """从 SMILES 预测目标性质。

        Args:
            smiles: SMILES 字符串或字符串列表。
            return_features: 是否同时返回 522-dim 特征矩阵。

        Returns:
            * 单分子, return_features=False → float
            * 单分子, return_features=True  → (float, np.ndarray)
            * 多分子, return_features=False → np.ndarray
            * 多分子, return_features=True  → (np.ndarray, np.ndarray)
        """
        single = isinstance(smiles, str)
        smiles_list = [smiles] if single else list(smiles)

        feats = []
        bad_indices = []
        for i, s in enumerate(smiles_list):
            f = _featurize_single(s)
            if f is None:
                bad_indices.append(i)
                f = np.full(522, np.nan, dtype=np.float32)
            feats.append(f)

        X = np.array(feats, dtype=np.float32)

        if single and bad_indices:
            raise ValueError(f"Invalid SMILES: '{smiles}'")

        if bad_indices:
            X[bad_indices] = 0.0
            print(f"[SurfPredict] Warning: {len(bad_indices)}/{len(smiles_list)} "
                  f"SMILES invalid; predictions may be unreliable.")

        preds_scaled = self.predict_from_features(X)
        # 还原 y_scale（Gamma_max 自动除 1e6，其他 target 除 1 = 不变）
        preds = preds_scaled / self.y_scale

        out = float(preds[0]) if single else preds
        if return_features:
            return out, X
        return out

    def predict_from_features(self, features: np.ndarray) -> np.ndarray:
        """从预计算的 (N, 522) 特征矩阵预测（返回缩放空间的值）。"""
        X = np.atleast_2d(np.asarray(features, dtype=np.float32))

        if self.model_name == 'all':
            return self._predict_ensemble(X)

        if self._model is None:
            raise RuntimeError("No model loaded. Call load() first or specify "
                               "model_name at init.")

        return self._predict_with(self._model, X)

    def _predict_with(self, model, X: np.ndarray) -> np.ndarray:
        """Run prediction with a single model (sklearn or torch)."""
        if _is_torch_model(model):
            with torch.no_grad():
                t = torch.tensor(X, dtype=torch.float32).to(self.device)
                return model(t).cpu().numpy()
        return model.predict(X)

    def _predict_ensemble(self, X: np.ndarray) -> np.ndarray:
        """Average predictions from all loaded models."""
        if not self._models:
            raise RuntimeError("No models loaded for ensemble.")
        preds = []
        for name, model in self._models.items():
            try:
                preds.append(self._predict_with(model, X))
            except Exception as e:
                print(f"  [SurfPredict] {name} failed: {e}")
        if not preds:
            raise RuntimeError("All ensemble models failed.")
        return np.mean(preds, axis=0)

    def __repr__(self):
        if self.model_name == 'all':
            return f"<SmilesPredictor ensemble ({self.target}, {len(self._models)} models)>"
        base = os.path.basename(self.run_dir) if self.run_dir else '?'
        return f"<SmilesPredictor {self.model_name} [{base}] target={self.target}>"


# ============================================================================
# Module-level convenience functions
# ============================================================================

def _resolve_run_dir(run_dir_from_index: str, target: Optional[str] = None) -> Optional[str]:
    """解析索引中的 run_dir 为实际存在的模型目录路径。

    兼容新旧两种路径格式：
      旧: runs\\catboost_20260722_181638         (文件已移至 runs/pCMC/ 下)
      新: runs/pCMC/pCMC_catboost_20260722_...   (直接可用)
    """
    # 1) 尝试直接拼接 PROJECT_ROOT
    d = os.path.join(PROJECT_ROOT, run_dir_from_index)
    if os.path.isfile(os.path.join(d, 'model.pkl')):
        return d

    # 2) 尝试加上 target 子目录
    #    runs\\catboost_20260722_181638 → runs/pCMC/catboost_20260722_181638
    if target:
        basename = os.path.basename(run_dir_from_index)
        d2 = os.path.join(PROJECT_ROOT, 'runs', target, basename)
        if os.path.isfile(os.path.join(d2, 'model.pkl')):
            return d2

    # 3) 尝试在 runs/{target}/ 下找目录名包含 target 前缀的
    #    runs/pCMC/pCMC_catboost_20260722_...
    if target:
        target_dir = os.path.join(PROJECT_ROOT, 'runs', target)
        if os.path.isdir(target_dir):
            basename = os.path.basename(run_dir_from_index)
            for entry in os.listdir(target_dir):
                if basename in entry and os.path.isfile(os.path.join(target_dir, entry, 'model.pkl')):
                    return os.path.join(target_dir, entry)

    return None


def list_models(target: Optional[str] = None) -> pd.DataFrame:
    """列出所有训练的模型及其性能指标。

    Args:
        target: 预测目标过滤，可选 'pCMC', 'AW_ST_CMC', 等。
                为 None 时读取全局索引（含所有 target）。

    Returns:
        DataFrame，按 test_rmse 升序排列。
    """
    if target is not None:
        idx_path = os.path.join(RUNS_DIR, target, '_runs_index.csv')
    else:
        idx_path = INDEX_PATH

    if not os.path.isfile(idx_path):
        if target:
            print(f"[SurfPredict] Index file not found: {idx_path}")
        else:
            raise FileNotFoundError(
                f"Index file not found: {idx_path}\n"
                "Run a training script first to generate model files."
            )
        return pd.DataFrame()

    with open(idx_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader]

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    for col in ['test_rmse', 'test_mae', 'test_r2', 'best_cv_rmse']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 只保留 model.pkl 仍存在的行（兼容新旧路径格式）
    valid = []
    for _, row in df.iterrows():
        resolved = _resolve_run_dir(row['run_dir'], target)
        if resolved:
            valid.append(row)

    if not valid:
        return pd.DataFrame()

    result = pd.DataFrame(valid)
    if 'test_rmse' in result.columns:
        result = result.sort_values('test_rmse', ascending=True).reset_index(drop=True)
    return result


def quick_predict(smiles: str, model_name: str = 'best',
                  target: str = 'pCMC',
                  device: str = 'cpu') -> float:
    """快速单分子预测。

    Examples:
        >>> quick_predict('CCO')
        >>> quick_predict('CCO', target='AW_ST_CMC')
        >>> quick_predict('CCO', model_name='catboost', target='Gamma_max')
    """
    return SmilesPredictor(model_name=model_name, target=target,
                           device=device).predict(smiles)


def SmilesPredict(smiles,
                  model_name: str = 'best',
                  target: str = 'pCMC',
                  run_dir: Optional[str] = None,
                  return_features: bool = False):
    """最简一行预测接口。

    Args:
        smiles: SMILES 字符串或列表。
        model_name: 模型名，默认 'best'（自动选当前 target 最优模型）。
        target: 预测目标，可选 'pCMC', 'AW_ST_CMC', 'Gamma_max',
                'Area_min', 'Pi_CMC', 'pC20'。默认 'pCMC'。
        run_dir: 指定运行目录（优先级最高）。
        return_features: 是否同时返回 522-dim 特征向量。

    Returns:
        float, np.ndarray, 或 (预测值, 特征矩阵) 元组。

    Examples:
        >>> SmilesPredict('CCO')                               # pCMC
        >>> SmilesPredict('CCO', target='AW_ST_CMC')            # 表面张力
        >>> SmilesPredict(['CCO', 'CCC(=O)O'], target='Gamma_max')  # 批量
        >>> SmilesPredict('CCO', model_name='catboost', target='pC20')
        >>> SmilesPredict('CCO', target='pCMC', return_features=True)
    """
    engine = SmilesPredictor(model_name=model_name, target=target,
                             run_dir=run_dir)
    return engine.predict(smiles, return_features=return_features)


# ============================================================================
# CLI (``python use/use_models.py --smiles "CCO" --target pCMC``)
# ============================================================================

def _cli():
    import argparse
    parser = argparse.ArgumentParser(
        description='SurfPredict — Predict surfactant properties from SMILES',
    )
    parser.add_argument('--smiles', '-s', nargs='+', required=True,
                        help='SMILES string(s) to predict')
    parser.add_argument('--model', '-m', default='best',
                        help='Model name or "best" (default)')
    parser.add_argument('--target', '-t', default='pCMC',
                        choices=sorted(TARGETS),
                        help='Target property (default: pCMC)')
    parser.add_argument('--list', '-l', nargs='?', const=None, default=False,
                        help='List models (optionally filter by target, e.g. -l pCMC)')
    parser.add_argument('--device', default='cpu',
                        help='Torch device (cpu/cuda)')
    args = parser.parse_args()

    if args.list is not False:
        target_arg = args.list if args.list else None
        print(list_models(target=target_arg))
        return

    pred = SmilesPredictor(model_name=args.model, target=args.target,
                           device=args.device)
    for s in args.smiles:
        p = pred.predict(s)
        print(f'{s}\t{p:.6f}')


if __name__ == '__main__':
    _cli()
