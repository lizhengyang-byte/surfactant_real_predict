"""
train_cif_use_pharmhgt_features.py — Conditional Inference Forest with PharmHGT-style Featurization
==================================================================================================

Uses shared featurization from smiles_to_features_pharmhgt.py (522-dim).
Features are cached under data/features/surfpro/ after first computation.

Conditional Inference Forest: unbiased recursive partitioning based on statistical
permutation tests (Hothorn et al., 2006). 在 Python 中使用 ExtraTreesRegressor 作为近似实现：
- 分裂变量选择使用显著性检验（近似：ExtraTrees 的随机阈值降低选择偏差）
- 树生长不受分裂变量偏倚影响（对高基数特征不会偏好）
- 每棵树基于不同的随机阈值子集训练，进一步降低方差

ExtraTreesRegressor details:
- 分裂时不仅随机选择特征，还随机选择分裂阈值
- 在保持与 RandomForest 相似精度的同时，进一步降低过拟合
- 高维特征上比 RandomForest 更有优势

Usage:
  python train_cif_use_pharmhgt_features.py

Data:
  ./data/surfpro/surfpro_train.csv  (training)
  ./data/surfpro/surfpro_test.csv   (test)
"""

import os, sys, math, random, warnings

import numpy as np
import pandas as pd

# Shared featurization
from smiles_to_features_pharmhgt import load_or_compute_features, FEATURE_NAMES
from utils import setup_run, save_metrics, update_index

# ExtraTreesRegressor — Python 中最接近 Conditional Inference Forest 的实现
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Optuna
import optuna
from optuna.pruners import MedianPruner

warnings.filterwarnings('ignore')

# ===========================================================================
# 预调优参数（跳过 Optuna，直接训练）
# 从一次成功的 Optuna 运行中获取 best trial params 填入此处。
# 设为 None 则执行完整的 Optuna 搜索。
# ===========================================================================
PRETUNED_PARAMS = None


# ===========================================================================
# Main — Load Data, Featurize, Train Conditional Inference Forest with Optuna
# ===========================================================================

def main():
    DATA_TRAIN = './data/surfpro/surfpro_train.csv'
    DATA_TEST = './data/surfpro/surfpro_test.csv'
    TARGET_COL = 'Area_min'
    SMILES_COL = 'SMILES'
    VAL_FRAC = 0.125
    SEED = 42
    N_OPTUNA_TRIALS = 50
    N_FOLDS = 5

    random.seed(SEED)
    np.random.seed(SEED)
    # ---- 初始化运行日志 ----
    run_dir = setup_run('cif', {
        'model': 'ConditionalInferenceForest',
        'feature_type': 'pharmhgt_522',
        'feature_dim': 522,
        'data_train': DATA_TRAIN,
        'data_test': DATA_TEST,
        'target_col': TARGET_COL,
        'val_frac': VAL_FRAC,
        'seed': SEED,
        'n_optuna_trials': N_OPTUNA_TRIALS,
        'n_folds': N_FOLDS,
    })

    print("=" * 60)
    print("Conditional Inference Forest + PharmHGT-style Featurization for Area_min Prediction")
    print("=" * 60)

    # ---- Load / featurize (cached) ----
    X_full, y_full, X_test, y_test = load_or_compute_features(
        train_csv=DATA_TRAIN, test_csv=DATA_TEST,
        target_col=TARGET_COL, smiles_col=SMILES_COL,
    )
    print(f"  Train features: {X_full.shape}")
    print(f"  Test features:  {X_test.shape}")

    # ---- Train/Validation split ----
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=VAL_FRAC, random_state=SEED)
    print(f"\nSplit: Train {len(X_train)}, Val {len(X_val)}, Test {len(X_test)}")

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
    else:
        print("\n" + "=" * 60)
        print(f"Optuna Hyperparameter Tuning ({N_OPTUNA_TRIALS} trials, {N_FOLDS}-Fold CV)")
        print("=" * 60)

        FEATURE_NAME = 'pharmhgt_522'

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 2000),
                'max_depth': trial.suggest_int('max_depth', 3, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 50),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 50),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'random_state': SEED,
                'n_jobs': -1,
                'verbose': 0,
            }

            # max_samples only valid when bootstrap=True
            if params['bootstrap']:
                params['max_samples'] = trial.suggest_float('max_samples', 0.5, 1.0)

            cv_scores = []
            kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
            for fold, (train_idx_cv, val_idx_cv) in enumerate(kf.split(X_full)):
                X_tr_cv = X_full[train_idx_cv]
                y_tr_cv = y_full[train_idx_cv]
                X_val_cv = X_full[val_idx_cv]
                y_val_cv = y_full[val_idx_cv]

                model_cv = ExtraTreesRegressor(**params)
                model_cv.fit(X_tr_cv, y_tr_cv)
                y_pred_cv = model_cv.predict(X_val_cv)
                rmse_cv = np.sqrt(mean_squared_error(y_val_cv, y_pred_cv))
                cv_scores.append(rmse_cv)

                mean_so_far = np.mean(cv_scores)
                trial.report(mean_so_far, fold)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return np.mean(cv_scores)

        sampler = optuna.samplers.TPESampler(seed=SEED)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=1, n_min_trials=3)
        study = optuna.create_study(
            study_name=f'cif_{FEATURE_NAME}',
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

    # ======================================================================
    # Final Training with Best Params
    # ======================================================================
    print("\n" + "=" * 60)
    print("Training Final Model with Best Hyperparameters")
    print("=" * 60)

    # 固定 seed 和并行
    best_params['random_state'] = SEED
    best_params['n_jobs'] = -1
    best_params['verbose'] = 0

    final_model = ExtraTreesRegressor(**best_params)
    final_model.fit(X_full, y_full)

    # ======================================================================
    # Evaluation
    # ======================================================================
    print(f"\n{'='*60}")
    print("Test Evaluation")
    print(f"{'='*60}")

    y_pred = final_model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)

    print(f"  Test MSE:  {test_mse:.4f}")
    print(f"  Test RMSE: {test_rmse:.4f}")
    print(f"  Test MAE:  {test_mae:.4f}")
    print(f"  Test R²:   {test_r2:.4f}")

    # ---- Feature Importance ----
    print(f"\n{'='*60}")
    print("Top 20 Feature Importances")
    print(f"{'='*60}")
    importances = final_model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:20]

    names = FEATURE_NAMES  # from smiles_to_features_pharmhgt

    for rank, idx in enumerate(top_idx):
        print(f"  {rank+1:2d}. {names[idx]:25s}  {importances[idx]:.1f}")

    # ---- Save predictions plot ----
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Conditional Inference Forest + PharmHGT Features - Area_min Prediction', fontsize=14)

        # Pred vs True
        ax = axes[0]
        ax.scatter(y_test, y_pred, alpha=0.6, edgecolors='k', linewidth=0.5)
        lims = [min(y_test.min(), y_pred.min()) - 0.5, max(y_test.max(), y_pred.max()) + 0.5]
        ax.plot(lims, lims, 'r--', alpha=0.8, linewidth=1)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel('True Area_min'); ax.set_ylabel('Predicted Area_min')
        ax.set_title(f'Test R² = {test_r2:.4f}')
        ax.axis('square')

        # Residuals
        ax = axes[1]
        residuals = y_test - y_pred
        ax.scatter(y_pred, residuals, alpha=0.6, edgecolors='k', linewidth=0.5)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.8)
        ax.set_xlabel('Predicted Area_min'); ax.set_ylabel('Residuals')
        ax.set_title(f'MAE = {test_mae:.4f}')

        plt.tight_layout()
        plot_path = os.path.join(run_dir, 'pred_vs_true.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to {plot_path}")
    except ImportError:
        print("\n(Matplotlib not available - skipping plot)")

    # ---- Save model ----
    import joblib
    model_path = os.path.join(run_dir, 'model.pkl')
    joblib.dump(final_model, model_path)
    print(f"Model saved to {model_path}")

    print(f"\n{'='*60}")
    print("SUMMARY — Conditional Inference Forest + PharmHGT Features")
    print(f"{'='*60}")
    print(f"  Features:  {X_full.shape[1]}-dim (atom_agg + bond_agg + MACCS + BRICS + surfactant + descriptors)")
    print(f"  Train:     {len(X_full)} (split {len(X_train)} train + {len(X_val)} val)")
    print(f"  Test:      {len(X_test)}")
    if study_best_value is not None:
        print(f"  Optuna:    {N_OPTUNA_TRIALS} trials, {N_FOLDS}-fold CV")
        print(f"  Best CV RMSE: {study_best_value:.4f}")
    else:
        print(f"  Params:    pretuned (skipped Optuna)")
    print(f"  Test RMSE: {test_rmse:.4f}")
    print(f"  Test MAE:  {test_mae:.4f}")
    print(f"  Test R²:   {test_r2:.4f}")

    # ---- 保存指标 & 更新索引 ----
    metrics = {
        'test_rmse': round(test_rmse, 4),
        'test_mae': round(test_mae, 4),
        'test_r2': round(test_r2, 4),
    }
    if study_best_value is not None:
        metrics['best_cv_rmse'] = round(study_best_value, 4)
    save_metrics(run_dir, metrics)
    update_index(run_dir, 'cif', metrics)


if __name__ == '__main__':
    main()
