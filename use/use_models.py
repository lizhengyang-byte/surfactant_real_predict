"""
SurfPredict — Unified Model Prediction API
===========================================

从 runs/ 自动加载已训练的模型，提供统一的分子性质预测接口。
可在项目内任意位置（包括其他子目录）通过 import 调用。

用法:
    from use.use_models import SmilesPredict, SmilesPredictor, list_models

    # ---- 方式 0：一行预测（最简单） ----
    pred = SmilesPredict('CCO')
    pred = SmilesPredict('CCO', model_name='catboost')
    pred = SmilesPredict(['CCO', 'CCC(=O)O'])          # 批量
    pred, feats = SmilesPredict('CCO', return_features=True)  # +特征向量

    # ---- 方式 1：自动选择最佳模型 ----
    predictor = SmilesPredictor()
    pred = predictor.predict('CCO')

    # ---- 方式 2：指定模型名称 ----
    predictor = SmilesPredictor(model_name='catboost')
    predictor = SmilesPredictor(model_name='xgboost')
    predictor = SmilesPredictor(model_name='mlp')

    # ---- 方式 3：指定具体运行目录 ----
    predictor = SmilesPredictor(run_dir='runs/xgboost_20260722_193605')

    # ---- 方式 4：加载所有模型做集成预测 ----
    predictor = SmilesPredictor(model_name='all')

    # ---- 方式 5：批量预测 + 特征向量 ----
    preds = predictor.predict(['CCO', 'CCC(=O)O', 'c1ccccc1'])
    pred, features = predictor.predict('CCO', return_features=True)
    print(features.shape)  # (522,)

    # ---- 查看可用模型 ----
    df = list_models()
"""

import csv
import os, re, warnings, sys, math
from typing import Optional

import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings('ignore')

# ============================================================================
# 将项目根目录加入 sys.path，确保从任何子目录都能导入同级的模块
# ============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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
            # input_dim stored in checkpoint metadata;
            # LSTM uses fixed input_size=1 (522 time steps × 1 feature).
            # activation is not configurable for LSTM, kept for compat.
            self.lstm = nn.LSTM(
                input_size=1,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                batch_first=True,
                dropout=dropout if n_layers > 1 else 0,
            )
            self.fc = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            x = x.unsqueeze(-1)          # (batch, 522, 1)
            lstm_out, _ = self.lstm(x)   # (batch, 522, hidden_dim)
            last_out = lstm_out[:, -1, :]  # (batch, hidden_dim)
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
        """Transformer Encoder regression network
        (matches train_transformer_use_pharmhgt_features.py)."""

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
            x = x.unsqueeze(-1)                     # (batch, 522, 1)
            x = self.input_proj(x)                  # (batch, 522, d_model)
            x = self.pos_encoder(x)
            x = self.transformer_encoder(x)         # (batch, 522, d_model)
            x = x.mean(dim=1)                       # (batch, d_model)
            return self.fc(x).squeeze(-1)


# ============================================================================
# Internal helpers
# ============================================================================

_RUN_DIR_PATTERN = re.compile(r'^(\w+)_(\d{8}_\d{6})$')

#: Names of tree-based (sklearn-compatible) models
TREE_MODEL_NAMES = {'catboost', 'lightgbm', 'xgboost', 'histgb', 'rf', 'ngboost', 'cif'}
#: Names of deep learning (PyTorch) models
DEEP_MODEL_NAMES = {'mlp', 'rnn', 'transformer'}


def _discover_model_runs():
    """Scan runs/ for available model runs.

    Returns:
        dict: {model_name: [(timestamp_str, run_dir_path), ...]},
              sorted newest-first per model. Skips dirs without model.pkl.
    """
    if not os.path.isdir(RUNS_DIR):
        raise FileNotFoundError(f"Runs directory not found: {RUNS_DIR}")

    models = {}
    for entry in os.listdir(RUNS_DIR):
        dirpath = os.path.join(RUNS_DIR, entry)
        if not os.path.isdir(dirpath):
            continue
        match = _RUN_DIR_PATTERN.match(entry)
        if not match:
            continue
        model_name = match.group(1)
        # Skip directories with no model.pkl (e.g. mlp_tuning temp dirs)
        if not os.path.isfile(os.path.join(dirpath, 'model.pkl')):
            continue
        models.setdefault(model_name, []).append((match.group(2), dirpath))

    for name in models:
        models[name].sort(key=lambda x: x[0], reverse=True)
    return models


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

    # Try PyTorch first
    if TORCH_AVAILABLE:
        try:
            ckpt = torch.load(model_path, map_location='cpu', weights_only=True)
            if 'model_type' in ckpt and 'state_dict' in ckpt:
                return _load_torch_model(run_dir, device)
        except Exception:
            pass

    # Fall back to sklearn/joblib
    return _load_sklearn_model(run_dir)


def _is_torch_model(model) -> bool:
    """Check if a loaded model is a PyTorch nn.Module."""
    return TORCH_AVAILABLE and isinstance(model, nn.Module)


# ============================================================================
# Public API
# ============================================================================

class SmilesPredictor:
    """Unified molecular property prediction interface.

    Automatically manages model loading, featurization, and prediction.

    Examples:
        >>> predictor = SmilesPredictor()
        >>> predictor.predict('CCO')
        6.323...

        >>> predictor = SmilesPredictor(model_name='catboost')
        >>> predictor.predict(['CCO', 'CCC(=O)O'])
        array([6.323, 4.012])

        >>> predictor = SmilesPredictor(model_name='all')  # ensemble
        >>> predictor.predict('CCO')
        6.152...
    """

    def __init__(self, model_name: str = 'best',
                 run_dir: Optional[str] = None,
                 device: str = 'cpu'):
        """
        Args:
            model_name: One of:
                - ``'best'``: Auto-select model with lowest test_rmse from index.
                - A model name: ``'catboost'``, ``'xgboost'``, ``'mlp'``,
                  ``'rnn'``, ``'transformer'``, ``'lightgbm'``, ``'histgb'``,
                  ``'rf'``, ``'ngboost'``, ``'cif'``.
                - ``'all'``: Load every available model and ensemble (mean).
            run_dir: Specific run directory path (overrides ``model_name``).
            device: Torch device string (only used for deep models).
        """
        self.device = device
        self.model_name = model_name
        self.run_dir = run_dir
        self._model = None        # single model (sklearn or torch)
        self._models = {}         # name -> model  (for 'all' mode)

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
        """Return a DataFrame with metadata for every trained model.

        Columns include ``model``, ``run_dir``, ``test_rmse``, ``test_mae``,
        ``test_r2``, and ``best_cv_rmse`` where available.  Sorted by
        test_rmse (best first).
        """
        return list_models()

    # ---- Loading -----------------------------------------------------------

    def load_best(self):
        """Auto-load the model with the lowest test_rmse from the runs index."""
        df = self.available_models()
        if df.empty:
            raise RuntimeError(
                "No trained models found.  Run a training script first:\n"
                "  python train_catboost_use_pharmhgt_features.py"
            )
        best = df.iloc[0]
        name = best['model']
        dir_rel = best['run_dir']
        dir_abs = os.path.join(PROJECT_ROOT, dir_rel)
        rmse = best.get('test_rmse', 'N/A')

        print(f"[SurfPredict] Best model: {name}  (test_rmse={rmse})")
        self.model_name = name
        self.run_dir = dir_abs
        self._load_from_dir(dir_abs)

    def load(self, model_name: str):
        """Load the latest run of a specific model.

        Args:
            model_name: e.g. 'catboost', 'xgboost', 'mlp', 'rnn', 'transformer'
        """
        models = _discover_model_runs()
        if model_name not in models:
            raise KeyError(
                f"Model '{model_name}' not found. "
                f"Available: {sorted(models)}\n"
                f"Use list_models() for full details."
            )
        ts, dirpath = models[model_name][0]
        print(f"[SurfPredict] Loading {model_name} (latest: {ts})")
        self.model_name = model_name
        self.run_dir = dirpath
        self._load_from_dir(dirpath)

    def load_from_dir(self, run_dir: str):
        """Load a model from a specific run directory."""
        self.run_dir = os.path.abspath(run_dir)
        self._load_from_dir(self.run_dir)
        m = _RUN_DIR_PATTERN.match(os.path.basename(self.run_dir))
        if m:
            self.model_name = m.group(1)

    def load_all(self):
        """Load every available model for ensemble prediction."""
        models = _discover_model_runs()
        if not models:
            raise RuntimeError("No trained models found.")

        loaded = 0
        for name in models:
            _, dirpath = models[name][0]
            try:
                self._models[name] = _load_single_model(dirpath, self.device)
                loaded += 1
            except Exception as e:
                print(f"  [SurfPredict] Failed to load {name}: {e}")

        self.model_name = 'all'
        print(f"[SurfPredict] Ensemble: {loaded}/{len(models)} models loaded")

    def _load_from_dir(self, run_dir):
        """(internal) Load model from a directory and store it."""
        self._model = _load_single_model(run_dir, self.device)

    # ---- Prediction --------------------------------------------------------

    def predict(self, smiles, return_features: bool = False):
        """Predict pCMC from SMILES string(s).

        Args:
            smiles: A SMILES string or an iterable of SMILES strings.
            return_features: If True, also return the 522-dim feature matrix.

        Returns:
            * Single SMILES, ``return_features=False`` → ``float``
            * Single SMILES, ``return_features=True``  → ``(float, np.ndarray)``
            * Multiple SMILES, ``return_features=False`` → ``np.ndarray``
            * Multiple SMILES, ``return_features=True``  → ``(np.ndarray, np.ndarray)``
        """
        single = isinstance(smiles, str)
        smiles_list = [smiles] if single else list(smiles)

        # Featurize each molecule
        feats = []
        bad_indices = []
        for i, s in enumerate(smiles_list):
            f = _featurize_single(s)
            if f is None:
                bad_indices.append(i)
                f = np.full(522, np.nan, dtype=np.float32)
            feats.append(f)

        X = np.array(feats, dtype=np.float32)

        # For single SMILES: raise immediately if invalid
        if single and bad_indices:
            raise ValueError(f"Invalid SMILES: '{smiles}'")

        # For batch: warn and fill invalid rows with NaN
        if bad_indices:
            X[bad_indices] = 0.0  # zero features so model doesn't crash
            print(f"[SurfPredict] Warning: {len(bad_indices)}/{len(smiles_list)} "
                  f"SMILES invalid; predictions may be unreliable.")

        preds = self.predict_from_features(X)

        out = float(preds[0]) if single else preds
        if return_features:
            return out, X
        return out

    def predict_from_features(self, features: np.ndarray) -> np.ndarray:
        """Predict from a pre-computed (N, 522) feature matrix.

        Args:
            features: shape ``(N, 522)`` or ``(522,)``.

        Returns:
            Predictions as a 1-D ``np.ndarray`` of length N.
        """
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
            return f"<SmilesPredictor ensemble ({len(self._models)} models)>"
        base = os.path.basename(self.run_dir) if self.run_dir else '?'
        return f"<SmilesPredictor {self.model_name} [{base}]>"


# ============================================================================
# Module-level convenience functions
# ============================================================================

def list_models() -> pd.DataFrame:
    """List all trained models with performance metrics.

    Returns:
        DataFrame with columns: model, run_dir, timestamp, test_rmse,
        test_mae, test_r2, best_cv_rmse (sorted by test_rmse ascending).
    """
    if not os.path.isfile(INDEX_PATH):
        raise FileNotFoundError(
            f"Index file not found: {INDEX_PATH}\n"
            "Run a training script first to generate model files."
        )
    with open(INDEX_PATH, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader]

    if not rows:
        print("[SurfPredict] Index file is empty.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Convert numeric columns (best-effort)
    for col in ['test_rmse', 'test_mae', 'test_r2', 'best_cv_rmse']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Keep only rows whose run_dir still has model.pkl
    valid = []
    for _, row in df.iterrows():
        d = os.path.join(PROJECT_ROOT, row['run_dir'])
        if os.path.isfile(os.path.join(d, 'model.pkl')):
            valid.append(row)

    if not valid:
        print("[SurfPredict] Warning: index entries exist but no model.pkl "
              "files found on disk.")
        return pd.DataFrame()

    result = pd.DataFrame(valid)
    if 'test_rmse' in result.columns:
        result = result.sort_values('test_rmse', ascending=True).reset_index(drop=True)
    return result


def quick_predict(smiles: str, model_name: str = 'best',
                  device: str = 'cpu') -> float:
    """Quick single-molecule prediction without creating a Predictor object.

    Examples:
        >>> quick_predict('CCO')
        6.323...
        >>> quick_predict('CCO', model_name='catboost')
        6.152...
    """
    return SmilesPredictor(model_name=model_name, device=device).predict(smiles)


def SmilesPredict(smiles, model_name='mlp', run_dir=None, return_features=False):
    """Simplified one-line prediction. 直接出结果，不创建对象。

    Args:
        smiles: SMILES 字符串或列表。
        model_name: 模型名，默认 'mlp'（当前最优）。
        run_dir: 指定运行目录（优先级高于 model_name）。
        return_features: 是否同时返回 522-dim 特征向量。

    Returns:
        float, np.ndarray, 或 (预测值, 特征矩阵) 元组。

    Examples:
        >>> SmilesPredict('CCO')
        0.1069
        >>> SmilesPredict(['CCO', 'CCC(=O)O'])
        array([0.1069, 0.0766])
        >>> SmilesPredict('CCO', model_name='catboost')
        0.9682
    """
    engine = SmilesPredictor(model_name=model_name, run_dir=run_dir)
    return engine.predict(smiles, return_features=return_features)


# ============================================================================
# CLI (``python use/use_models.py --smiles "CCO"``)
# ============================================================================

def _cli():
    import argparse
    parser = argparse.ArgumentParser(
        description='SurfPredict — Predict surfactant pCMC from SMILES',
    )
    parser.add_argument('--smiles', '-s', nargs='+', required=True,
                        help='SMILES string(s) to predict')
    parser.add_argument('--model', '-m', default='best',
                        help='Model name or "best" (default)')
    parser.add_argument('--list', '-l', action='store_true',
                        help='List available models and exit')
    parser.add_argument('--device', default='cpu',
                        help='Torch device (cpu/cuda)')
    args = parser.parse_args()

    if args.list:
        print(list_models())
        return

    pred = SmilesPredictor(model_name=args.model, device=args.device)
    for s in args.smiles:
        p = pred.predict(s)
        print(f'{s}\t{p:.4f}')


if __name__ == '__main__':
    _cli()
