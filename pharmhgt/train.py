#!/usr/bin/env python
"""
Training pipeline for PharmHGT on surfactant property prediction.

Usage:
    python -m pharmhgt.train                     # default config
    python -m pharmhgt.train --target pCMC        # specify target
    python -m pharmhgt.train --epochs 500 --lr 1e-3
"""

import argparse, json, math, os, sys, warnings
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from pharmhgt.config import PharmHGTConfig
from pharmhgt.data import load_and_build_graphs, collate_graphs
from pharmhgt.model import PharmHGT

warnings.filterwarnings('ignore')


# ===========================================================================
# Noam-style learning rate scheduler
# ===========================================================================
class NoamLR(LambdaLR):
    def __init__(self, optimizer, warmup_steps, model_dim, factor=1.0):
        self.warmup = warmup_steps
        self.factor = factor
        self.model_dim = model_dim
        super().__init__(optimizer, self._lr_lambda)

    def _lr_lambda(self, step):
        step += 1
        return self.factor * (self.model_dim ** -0.5) * min(
            step ** -0.5, step * (self.warmup ** -1.5)
        )


# ===========================================================================
# Training utilities
# ===========================================================================
def train_epoch(model, loader, optimizer, criterion, device, clip=None):
    model.train()
    total_loss = 0.0
    for batch_g, labels in loader:
        batch_g = batch_g.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        pred = model(batch_g)
        loss = criterion(pred, labels)
        loss.backward()

        if clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        total_loss += loss.item() * labels.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for batch_g, labels in loader:
        batch_g = batch_g.to(device)
        labels = labels.to(device)

        pred = model(batch_g)
        loss = criterion(pred, labels)
        total_loss += loss.item() * labels.size(0)

        all_preds.append(pred.cpu().numpy())
        all_targets.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)

    return total_loss / len(loader.dataset), rmse, mae, r2


# ===========================================================================
# Main training function
# ===========================================================================
def train_model(cfg: PharmHGTConfig):
    device = torch.device(cfg.device)
    print(f"Device: {device}")
    print(f"Config: hid_dim={cfg.hid_dim}, depth={cfg.depth}, heads={cfg.num_heads}")
    print(f"         batch={cfg.batch_size}, lr={cfg.lr}, epochs={cfg.epochs}")

    seed = cfg.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)

    # ---- Load data ----
    train_csv = os.path.join(cfg.data_dir, cfg.train_csv)
    test_csv = os.path.join(cfg.data_dir, cfg.test_csv)

    print(f"\nLoading data...")
    print(f"  Train: {train_csv}")
    print(f"  Test:  {test_csv}")
    print(f"  Target: {cfg.target_col}")

    train_dataset, test_dataset = load_and_build_graphs(
        train_csv=train_csv,
        test_csv=test_csv,
        target_col=cfg.target_col,
        smiles_col=cfg.smiles_col,
        cache_dir=cfg.graph_cache_dir,
    )

    # ---- Train/Val split ----
    indices = list(range(len(train_dataset)))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.125, random_state=cfg.seed
    )

    train_subset = torch.utils.data.Subset(train_dataset, train_idx)
    val_subset = torch.utils.data.Subset(train_dataset, val_idx)

    train_loader = DataLoader(
        train_subset, batch_size=cfg.batch_size,
        shuffle=True, collate_fn=collate_graphs, num_workers=0
    )
    val_loader = DataLoader(
        val_subset, batch_size=cfg.batch_size,
        shuffle=False, collate_fn=collate_graphs, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size,
        shuffle=False, collate_fn=collate_graphs, num_workers=0
    )

    print(f"  Train: {len(train_subset)} | Val: {len(val_subset)} | Test: {len(test_dataset)}")

    # ---- Build model ----
    model = PharmHGT(
        atom_dim=cfg.atom_dim,
        bond_dim=cfg.bond_dim,
        pharm_dim=cfg.pharm_dim,
        reac_dim=cfg.reac_dim,
        hid_dim=cfg.hid_dim,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        dropout=cfg.dropout,
        num_task=cfg.num_task,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.MSELoss()

    steps_per_epoch = len(train_loader)
    total_steps = cfg.epochs * steps_per_epoch
    scheduler = NoamLR(optimizer, warmup_steps=cfg.warmup_epochs * steps_per_epoch,
                       model_dim=cfg.hid_dim, factor=1.0)

    # ---- Training loop ----
    best_val_rmse = float('inf')
    best_state = None
    trigger = 0
    metrics_history = []

    print(f"\n{'='*60}")
    print(f"Training started")
    print(f"{'='*60}")

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, cfg.gradient_clip
        )
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        if epoch % 5 == 0 or epoch == cfg.epochs:
            val_loss, val_rmse, val_mae, val_r2 = evaluate(
                model, val_loader, criterion, device
            )

            metrics_history.append({
                'epoch': epoch, 'train_loss': train_loss,
                'val_loss': val_loss, 'val_rmse': val_rmse,
                'val_mae': val_mae, 'val_r2': val_r2, 'lr': current_lr,
            })

            improved = val_rmse < best_val_rmse
            if improved:
                best_val_rmse = val_rmse
                best_state = deepcopy(model.state_dict())
                trigger = 0
            else:
                trigger += 1

            print(
                f"  Epoch {epoch:3d} | "
                f"Train loss: {train_loss:.4f} | "
                f"Val RMSE: {val_rmse:.4f} (best: {best_val_rmse:.4f}) | "
                f"LR: {current_lr:.2e}"
            )

            if trigger >= cfg.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    # ---- Restore best model ----
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)

    # ---- Test evaluation ----
    print(f"\n{'='*60}")
    print(f"Test Evaluation")
    print(f"{'='*60}")

    test_loss, test_rmse, test_mae, test_r2 = evaluate(
        model, test_loader, criterion, device
    )
    print(f"  Test RMSE: {test_rmse:.4f}")
    print(f"  Test MAE:  {test_mae:.4f}")
    print(f"  Test R²:   {test_r2:.4f}")

    # ---- Save model ----
    os.makedirs(cfg.save_dir, exist_ok=True)
    model_path = os.path.join(cfg.save_dir, 'pharmhgt_model.pt')
    torch.save({
        'model_config': {
            'atom_dim': cfg.atom_dim,
            'bond_dim': cfg.bond_dim,
            'pharm_dim': cfg.pharm_dim,
            'reac_dim': cfg.reac_dim,
            'hid_dim': cfg.hid_dim,
            'depth': cfg.depth,
            'num_heads': cfg.num_heads,
            'dropout': cfg.dropout,
            'num_task': cfg.num_task,
        },
        'state_dict': model.state_dict(),
        'val_rmse': best_val_rmse,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2,
    }, model_path)
    print(f"Model saved to {model_path}")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Model:     PharmHGT (hid_dim={cfg.hid_dim}, depth={cfg.depth})")
    print(f"  Target:    {cfg.target_col}")
    print(f"  Train:     {len(train_subset)} | Val: {len(val_subset)} | Test: {len(test_dataset)}")
    print(f"  Val RMSE:  {best_val_rmse:.4f}")
    print(f"  Test RMSE: {test_rmse:.4f}")
    print(f"  Test MAE:  {test_mae:.4f}")
    print(f"  Test R²:   {test_r2:.4f}")

    return metrics_history, (test_rmse, test_mae, test_r2)


# ===========================================================================
# Entry point
# ===========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', default='pCMC')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--hid_dim', type=int, default=None)
    parser.add_argument('--depth', type=int, default=None)
    args = parser.parse_args()

    cfg = PharmHGTConfig()
    if args.target:
        cfg.target_col = args.target
    if args.epochs:
        cfg.epochs = args.epochs
    if args.lr:
        cfg.lr = args.lr
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.hid_dim:
        cfg.hid_dim = args.hid_dim
    if args.depth:
        cfg.depth = args.depth

    train_model(cfg)


if __name__ == '__main__':
    main()
