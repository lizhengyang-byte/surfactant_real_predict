"""
train_mlp_use_pharmhgt_features.py — MLP with PharmHGT-style Featurization
=================================================================================

Uses shared featurization from smiles_to_features_pharmhgt.py (522-dim).
Features are cached under data/features/surfpro/ after first computation.

Usage:
  python train_mlp_use_pharmhgt_features.py

Data:
  ./data/surfpro/surfpro_train.csv  (training)
  ./data/surfpro/surfpro_test.csv   (test)
"""

import sys, math, random, warnings, os

import numpy as np
import pandas as pd

# Shared featurization
from smiles_to_features_pharmhgt import load_or_compute_features
from utils import setup_run, save_metrics, update_index

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Optuna
import optuna
from optuna.pruners import MedianPruner

warnings.filterwarnings('ignore')

# Detect device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ===========================================================================
# 预调优参数（跳过 Optuna，直接训练）
# ===========================================================================
PRETUNED_PARAMS = None
# PRETUNED_PARAMS = {
#     'n_layers': 4, 'hidden_dim': 512, 'dropout': 0.1,
#     'activation': 'gelu', 'lr': 1e-3, 'weight_decay': 1e-6,
#     'batch_size': 64, 'n_epochs': 3000,
# }


# ===========================================================================
# MLP Model Definition
# ===========================================================================

class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout, activation='relu'):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.1))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ===========================================================================
# Training helpers
# ===========================================================================

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        total_loss += loss.item() * X_batch.size(0)
        all_preds.append(pred.cpu().numpy())
        all_targets.append(y_batch.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    return total_loss / len(loader.dataset), rmse


def make_loader(X, y, batch_size, shuffle=True, drop_last=False):
    tensor_x = torch.tensor(X, dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


def train_and_eval_mlp(X_tr, y_tr, X_val, y_val, params, n_epochs, device, verbose=False):
    """Train MLP with given params and return best validation RMSE + state dict."""
    input_dim = X_tr.shape[1]
    model = MLPRegressor(
        input_dim=input_dim,
        hidden_dim=params['hidden_dim'],
        n_layers=params['n_layers'],
        dropout=params['dropout'],
        activation=params['activation'],
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    criterion = nn.MSELoss()

    tr_loader = make_loader(X_tr, y_tr, params['batch_size'], shuffle=True, drop_last=True)
    val_loader = make_loader(X_val, y_val, params['batch_size'], shuffle=False)

    best_rmse = float('inf')
    patience = 30
    trigger = 0
    best_state = None
    report_interval = max(n_epochs // 10, 10)

    for epoch in range(1, n_epochs + 1):
        train_epoch(model, tr_loader, optimizer, criterion, device)
        if epoch % 5 == 0 or epoch == n_epochs:
            _, val_rmse = evaluate(model, val_loader, criterion, device)
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                trigger = 0
            else:
                trigger += 1
            if verbose and epoch % report_interval == 0:
                print(f"    Epoch {epoch:3d}/{n_epochs} — Val RMSE: {val_rmse:.4f} (best: {best_rmse:.4f})")
            if trigger >= patience:
                if verbose:
                    print(f"    Early stopping at epoch {epoch}")
                break

    return best_rmse, best_state


# ===========================================================================
# 5. Main — Load Data, Featurize, Train MLP
# ===========================================================================

def main():
    DATA_TRAIN = './data/surfpro/surfpro_train.csv'
    DATA_TEST = './data/surfpro/surfpro_test.csv'
    TARGET_COL = 'AW_ST_CMC'
    SMILES_COL = 'SMILES'
    VAL_FRAC = 0.125
    SEED = 42
    N_OPTUNA_TRIALS = 50
    N_FOLDS = 5
    # CV 阶段用较少 epoch（快速搜索），最终训练充分收敛
    CV_EPOCHS = 300

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if DEVICE.type == 'cuda':
        torch.cuda.manual_seed(SEED)

    # ---- Load / featurize (cached) ----
    X_full, y_full, X_test, y_test = load_or_compute_features(
        train_csv=DATA_TRAIN, test_csv=DATA_TEST,
        target_col=TARGET_COL, smiles_col=SMILES_COL,
    )
    input_dim = X_full.shape[1]
    print(f"  Input dim: {input_dim}")

    # ---- Train/Validation split ----
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=VAL_FRAC, random_state=SEED)
    print(f"\nSplit: Train {len(X_train)}, Val {len(X_val)}, Test {len(X_test)}")

    # ======================================================================
    # 准备运行日志（拆分为两步：先创建 run_dir，等 Optuna 后有最佳参数再填充 config）
    # ======================================================================

    # ======================================================================
    # Hyperparameter Optimization (Optuna) 或使用预调优参数
    # ======================================================================
    if PRETUNED_PARAMS is not None:
        print("\n" + "=" * 60)
        print("Using pretuned hyperparameters (skipping Optuna search)")
        print("=" * 60)
        best_params = PRETUNED_PARAMS.copy()
        study_best_value = None
        print(f"  Params: {best_params}")
        run_dir = setup_run('mlp', {'model': 'mlp', **best_params})
    else:
        # Optuna 搜索完后才知道最终参数，先创建临时标签
        run_dir = setup_run('mlp_tuning', {'model': 'mlp', 'phase': 'optuna_search'})
        print(f"\n{'='*60}")
        print(f"Optuna Hyperparameter Tuning ({N_OPTUNA_TRIALS} trials, {N_FOLDS}-Fold CV)")
        print(f"{'='*60}")

        def objective(trial):
            params = {
                'n_layers': trial.suggest_int('n_layers', 2, 6),
                'hidden_dim': trial.suggest_int('hidden_dim', 128, 1024),
                'dropout': trial.suggest_float('dropout', 0.0, 0.5),
                'activation': trial.suggest_categorical('activation', ['relu', 'gelu', 'leaky_relu']),
                'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
                'weight_decay': trial.suggest_float('weight_decay', 1e-8, 1e-3, log=True),
                'batch_size': trial.suggest_int('batch_size', 16, 128),
            }

            cv_scores = []
            kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
            for fold, (train_idx_cv, val_idx_cv) in enumerate(kf.split(X_full)):
                X_tr_cv = X_full[train_idx_cv]
                y_tr_cv = y_full[train_idx_cv]
                X_val_cv = X_full[val_idx_cv]
                y_val_cv = y_full[val_idx_cv]

                val_rmse, _ = train_and_eval_mlp(
                    X_tr_cv, y_tr_cv, X_val_cv, y_val_cv,
                    params, n_epochs=CV_EPOCHS, device=DEVICE, verbose=False,
                )
                cv_scores.append(val_rmse)

                mean_so_far = np.mean(cv_scores)
                trial.report(mean_so_far, fold)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return np.mean(cv_scores)

        sampler = optuna.samplers.TPESampler(seed=SEED)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=1, n_min_trials=3)
        study = optuna.create_study(
            study_name='mlp_pharmhgt_522',
            direction='minimize',
            sampler=sampler,
            pruner=pruner,
        )
        study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)

        print(f"\n=== Best Trial ===")
        print(f"  CV RMSE: {study.best_value:.6f}")
        print(f"  Params:  {study.best_params}")

        best_params = study.best_params.copy()
        study_best_value = study.best_value

        # ---- 创建最终运行的日志（用最佳参数重新初始化 run_dir） ----
        final_config = {'model': 'mlp', 'optuna_trials': N_OPTUNA_TRIALS, 'n_folds': N_FOLDS, **best_params}
        final_config['best_cv_rmse'] = round(study_best_value, 4)
        run_dir = setup_run('mlp', final_config)

    print("=" * 60)
    print("MLP + PharmHGT-style Featurization for LogCMC AW_ST_CMC Prediction")
    print("=" * 60)

    # ======================================================================
    # Final Training with Best Params
    # ======================================================================
    print("\n" + "=" * 60)
    print("Training Final Model")
    print("=" * 60)

    n_epochs_full = best_params.pop('n_epochs', 3000)
    train_rmse, best_state = train_and_eval_mlp(
        X_full, y_full, X_val, y_val, best_params,
        n_epochs=n_epochs_full, device=DEVICE, verbose=True,
    )
    best_val_rmse = train_rmse  # 来自 train_and_eval_mlp 的返回值就是 best val RMSE

    # Build final model with best weights for evaluation
    final_model = MLPRegressor(
        input_dim=input_dim,
        hidden_dim=best_params['hidden_dim'],
        n_layers=best_params['n_layers'],
        dropout=best_params['dropout'],
        activation=best_params['activation'],
    ).to(DEVICE)
    if best_state is not None:
        final_model.load_state_dict(best_state)
    final_model.to(DEVICE)

    # ======================================================================
    # Evaluation
    # ======================================================================
    print(f"\n{'='*60}")
    print("Test Evaluation")
    print(f"{'='*60}")

    final_model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        y_pred = final_model(X_test_t).cpu().numpy()

    test_mse = mean_squared_error(y_test, y_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)

    print(f"  Test MSE:  {test_mse:.4f}")
    print(f"  Test RMSE: {test_rmse:.4f}")
    print(f"  Test MAE:  {test_mae:.4f}")
    print(f"  Test R²:   {test_r2:.4f}")

    # ---- Save model ----
    model_path = os.path.join(run_dir, 'model.pkl')
    torch.save({
        'model_type': 'mlp',
        'input_dim': input_dim,
        'n_layers': best_params['n_layers'],
        'hidden_dim': best_params['hidden_dim'],
        'dropout': best_params['dropout'],
        'activation': best_params['activation'],
        'state_dict': final_model.state_dict(),
    }, model_path)
    print(f"Model saved to {model_path}")

    print(f"\n{'='*60}")
    print("SUMMARY — MLP + PharmHGT Features")
    print(f"{'='*60}")
    print(f"  Features:  {input_dim}-dim (atom_agg + bond_agg + MACCS + BRICS + surfactant + descriptors)")
    print(f"  Train:     {len(X_full)} (split {len(X_train)} train + {len(X_val)} val)")
    print(f"  Test:      {len(X_test)}")
    if study_best_value is not None:
        print(f"  Optuna:    {N_OPTUNA_TRIALS} trials, {N_FOLDS}-fold CV")
        print(f"  Best CV RMSE: {study_best_value:.4f}")
    print(f"  Best val RMSE: {best_val_rmse:.4f}")
    print(f"  Test RMSE: {test_rmse:.4f}")
    print(f"  Test MAE:  {test_mae:.4f}")
    print(f"  Test R²:   {test_r2:.4f}")

    # ---- 保存指标 & 更新索引 ----
    metrics = {
        'test_rmse': round(test_rmse, 4),
        'test_mae': round(test_mae, 4),
        'test_r2': round(test_r2, 4),
        'best_val_rmse': round(best_val_rmse, 4),
    }
    if study_best_value is not None:
        metrics['best_cv_rmse'] = round(study_best_value, 4)
    save_metrics(run_dir, metrics)
    update_index(run_dir, 'mlp', metrics)


if __name__ == '__main__':
    main()
