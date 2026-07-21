"""
PharmHGT training with Optuna hyperparameter tuning.

Usage:
    python pharmhgt/test_example.py

To use your own data, change `data_path` below to point to a CSV
with at least 'SMILES' and 'pCMC' columns.
"""
import sys, os, json, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')

from pharmhgt.data import SurfactantGraphDataset, collate_graphs
from pharmhgt.model import PharmHGT

import optuna

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_TRIALS = 30
N_EPOCHS = 100
EARLY_STOP = 30


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_targets = [], []
    for batch_g, labels in loader:
        batch_g = batch_g.to(DEVICE)
        pred = model(batch_g)
        all_preds.append(pred.cpu().numpy())
        all_targets.append(labels.cpu().numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


def train_one_model(hid_dim, depth, num_heads, dropout, lr, weight_decay,
                    batch_size, train_loader, val_loader, verbose=True):
    model = PharmHGT(
        hid_dim=hid_dim, depth=depth, num_heads=num_heads,
        dropout=dropout, num_task=1,
    ).to(DEVICE)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS)

    best_val_rmse = float('inf')
    best_state = None
    trigger = 0

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch_g, labels in train_loader:
            batch_g, labels = batch_g.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            pred = model(batch_g)
            loss = criterion(pred, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item() * len(labels)
        scheduler.step()

        val_rmse, val_mae, val_r2 = evaluate(model, val_loader)
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            trigger = 0
        else:
            trigger += 1

        if trigger >= EARLY_STOP:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return best_val_rmse, model


def objective(trial, train_loader, val_loader):
    hid_dim = trial.suggest_categorical('hid_dim', [128, 256, 384, 512])
    depth = trial.suggest_int('depth', 2, 5)
    num_heads = trial.suggest_categorical('num_heads', [2, 4, 8])
    dropout = trial.suggest_float('dropout', 0.0, 0.25)
    lr = trial.suggest_float('lr', 5e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-7, 5e-5, log=True)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])

    train_loader.dataset.dataset._graphs = [None] * len(train_loader.dataset.dataset)
    sub_train = Subset(train_loader.dataset.dataset,
                       [train_loader.dataset.indices[i] for i in range(len(train_loader.dataset))])
    sub_val = Subset(val_loader.dataset.dataset,
                     [val_loader.dataset.indices[i] for i in range(len(val_loader.dataset))])

    tr_loader = DataLoader(sub_train, batch_size=batch_size, shuffle=True,
                           collate_fn=collate_graphs, num_workers=0)
    vl_loader = DataLoader(sub_val, batch_size=batch_size, shuffle=False,
                           collate_fn=collate_graphs, num_workers=0)

    val_rmse, _ = train_one_model(
        hid_dim, depth, num_heads, dropout, lr, weight_decay,
        batch_size, tr_loader, vl_loader, verbose=False,
    )
    return val_rmse


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
data_path = os.path.join(os.path.dirname(__file__), 'example_data.csv')
df = pd.read_csv(data_path)
df = df.dropna(subset=['SMILES', 'pCMC'])
print(f'Total molecules: {len(df)}')

# ---------------------------------------------------------------------------
# 2. Split: train 80%, val 10%, test 10%
# ---------------------------------------------------------------------------
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.125, random_state=42)

print(f'Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}')

# ---------------------------------------------------------------------------
# 3. Build datasets
# ---------------------------------------------------------------------------
print('Building graphs...')
full_dataset = SurfactantGraphDataset(
    pd.concat([train_df, val_df, test_df]),
    smiles_col='SMILES', target_col='pCMC', verbose=False,
)

n_train = len(train_df)
n_val = len(val_df)

train_dataset = Subset(full_dataset, list(range(n_train)))
val_dataset = Subset(full_dataset, list(range(n_train, n_train + n_val)))
test_dataset = Subset(full_dataset, list(range(n_train + n_val, len(full_dataset))))

print(f'  Valid molecules: {len(full_dataset)}')
print()

# ---------------------------------------------------------------------------
# 4. Optuna hyperparameter search
# ---------------------------------------------------------------------------
print('=' * 55)
print(f'Optuna Search ({N_TRIALS} trials)')
print('=' * 55)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,
                          collate_fn=collate_graphs, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False,
                        collate_fn=collate_graphs, num_workers=0)

study = optuna.create_study(direction='minimize',
                            sampler=optuna.samplers.TPESampler(seed=42),
                            pruner=optuna.pruners.MedianPruner())

study.optimize(lambda trial: objective(trial, train_loader, val_loader),
               n_trials=N_TRIALS, show_progress_bar=True)

print(f'\nBest trial: #{study.best_trial.number}')
print(f'Best val RMSE: {study.best_value:.4f}')
print('Best params:')
for k, v in study.best_params.items():
    print(f'  {k}: {v}')
print()

# ---------------------------------------------------------------------------
# 5. Retrain with best params on train+val, evaluate on test
# ---------------------------------------------------------------------------
print('=' * 55)
print('Final Training with Best Hyperparameters')
print('=' * 55)

bp = study.best_params
batch_size = bp['batch_size']

train_val_dataset = Subset(full_dataset, list(range(n_train + n_val)))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                         collate_fn=collate_graphs, num_workers=0)

train_val_loader = DataLoader(train_val_dataset, batch_size=batch_size,
                              shuffle=True, collate_fn=collate_graphs,
                              num_workers=0)
val_loader_final = DataLoader(val_dataset, batch_size=batch_size,
                              shuffle=False, collate_fn=collate_graphs,
                              num_workers=0)

val_rmse, model = train_one_model(
    bp['hid_dim'], bp['depth'], bp['num_heads'], bp['dropout'],
    bp['lr'], bp['weight_decay'], batch_size,
    train_val_loader, val_loader_final, verbose=True,
)

test_rmse, test_mae, test_r2 = evaluate(model, test_loader)

print(f'\n{"=" * 55}')
print(f'Test Results')
print(f'{"=" * 55}')
print(f'  Test RMSE: {test_rmse:.4f}')
print(f'  Test MAE:  {test_mae:.4f}')
print(f'  Test R2:   {test_r2:.4f}')
print(f'  Best params: {study.best_params}')

# ---- Save model & study ----
os.makedirs('models/pharmhgt', exist_ok=True)
torch.save({
    'best_params': study.best_params,
    'model_config': {
        'hid_dim': bp['hid_dim'], 'depth': bp['depth'],
        'num_heads': bp['num_heads'], 'dropout': bp['dropout'],
    },
    'state_dict': model.state_dict(),
    'test_rmse': test_rmse, 'test_mae': test_mae, 'test_r2': test_r2,
}, 'models/pharmhgt/pharmhgt_best.pt')

joblib_file = 'models/pharmhgt/optuna_study.pkl'
try:
    import joblib
    joblib.dump(study, joblib_file)
except Exception:
    pass

print(f'\nModel saved to models/pharmhgt/pharmhgt_best.pt')
print('Done.')
