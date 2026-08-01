"""
train_ngboost_use_pharmhgt_features.py — NGBoost with PharmHGT-style Featurization
=================================================================================

Uses shared featurization from smiles_to_features_pharmhgt.py (522-dim).
Features are cached under data/features/surfpro/ after first computation.

NGBoost: Natural Gradient Boosting for probabilistic prediction.
Outputs a Normal distribution (mean + variance) instead of a point estimate.

Usage:
  python train_ngboost_use_pharmhgt_features.py

Data:
  ./data/surfpro/surfpro_train.csv  (training)
  ./data/surfpro/surfpro_test.csv   (test)
"""

import os, random, warnings

import numpy as np
import pandas as pd

# Shared featurization
from train.train_AW_ST_CMCmodels.smiles_to_features_pharmhgt import load_or_compute_features, FEATURE_NAMES
from train.train_AW_ST_CMCmodels.utils import setup_run, save_metrics, update_index

# NGBoost
from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import LogScore, CRPScore
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor

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

PRETUNED_PARAMS = {
    'n_estimators': 637,
    'learning_rate': 0.03374093172394316,
    'minibatch_frac': 0.7231391767644131,
    'max_depth': 5,          # 会被提取到 base_learner
    'min_samples_leaf': 13,  # 会被提取到 base_learner
    'score_rule': 'LogScore',
}

# ===========================================================================
# Main — Load Data, Featurize, Train NGBoost with Optuna
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

    random.seed(SEED)
    np.random.seed(SEED)
    # ---- 初始化运行日志 ----
    run_dir = setup_run('ngboost', {
        'model': 'NGBoost',
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
    print("NGBoost + PharmHGT-style Featurization for AW_ST_CMC Prediction")
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
        best_params_all = PRETUNED_PARAMS.copy()
        study_best_value = None
        print(f"  Params: {best_params_all}")
    else:
        print("\n" + "=" * 60)
        print(f"Optuna Hyperparameter Tuning ({N_OPTUNA_TRIALS} trials, {N_FOLDS}-Fold CV)")
        print("=" * 60)

        FEATURE_NAME = 'pharmhgt_522'

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 2000),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.5, log=True),
                'minibatch_frac': trial.suggest_float('minibatch_frac', 0.3, 1.0),
                'verbose': False,
            }

            base_params = {
                'max_depth': trial.suggest_int('max_depth', 2, 8),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 100),
            }

            score_choice = trial.suggest_categorical('score_rule', ['LogScore', 'CRPScore'])

            cv_scores = []
            kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
            for fold, (train_idx_cv, val_idx_cv) in enumerate(kf.split(X_full)):
                X_tr_cv = X_full[train_idx_cv]
                y_tr_cv = y_full[train_idx_cv]
                X_val_cv = X_full[val_idx_cv]
                y_val_cv = y_full[val_idx_cv]

                base_learner = DecisionTreeRegressor(**base_params, random_state=SEED)
                Score = LogScore if score_choice == 'LogScore' else CRPScore

                model_cv = NGBRegressor(
                    Base=base_learner,
                    Dist=Normal,
                    Score=Score,
                    natural_gradient=True,
                    random_state=SEED,
                    **params,
                )
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
            study_name=f'ngboost_{FEATURE_NAME}',
            direction='minimize',
            sampler=sampler,
            pruner=pruner,
        )
        study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)

        print(f"\n=== Best Trial ===")
        print(f"  CV RMSE: {study.best_value:.6f}")
        print(f"  Params:  {study.best_params}")

        best_params_all = study.best_params.copy()
        study_best_value = study.best_value

    # ======================================================================
    # Final Training with Best Params
    # ======================================================================
    print("\n" + "=" * 60)
    print("Training Final Model with Best Hyperparameters")
    print("=" * 60)

    # Extract base learner and score params from the combined param dict
    base_params = {
        'max_depth': best_params_all.pop('max_depth', 5),
        'min_samples_leaf': best_params_all.pop('min_samples_leaf', 20),
    }
    score_name = best_params_all.pop('score_rule', 'LogScore')
    Score = LogScore if score_name == 'LogScore' else CRPScore

    base_learner = DecisionTreeRegressor(**base_params, random_state=SEED)
    final_model = NGBRegressor(
        Base=base_learner,
        Dist=Normal,
        Score=Score,
        natural_gradient=True,
        random_state=SEED,
        **best_params_all,
    )
    final_model.fit(X_full, y_full)

    # ======================================================================
    # Evaluation
    # ======================================================================
    print(f"\n{'='*60}")
    print("Test Evaluation")
    print(f"{'='*60}")

    y_pred = final_model.predict(X_test)
    y_dist = final_model.pred_dist(X_test)

    test_mse = mean_squared_error(y_test, y_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)

    print(f"  Test MSE:  {test_mse:.4f}")
    print(f"  Test RMSE: {test_rmse:.4f}")
    print(f"  Test MAE:  {test_mae:.4f}")
    print(f"  Test R²:   {test_r2:.4f}")
    print(f"  Distribution: Normal (mean ± std)")
    print(f"  Avg pred std: {np.mean(y_dist.std()):.4f}")

    # ---- Feature Importance ----
    # NGBoost 的 feature_importances_ 不是标准 1D 数组（可能为 2D 或嵌套结构）
    # 此处尝试提取，失败则跳过
    print(f"\n{'='*60}")
    print("Feature Importance")
    print(f"{'='*60}")
    try:
        imp_raw = final_model.feature_importances_
        # 尝试转为 1D 数组
        if hasattr(imp_raw, 'ravel'):
            imp_1d = np.ravel(imp_raw)
        else:
            imp_1d = np.asarray(imp_raw).ravel()
        if imp_1d.ndim == 1 and imp_1d.shape[0] == X_full.shape[1]:
            top_idx = np.argsort(imp_1d)[::-1][:20]
            names = FEATURE_NAMES
            for rank, idx in enumerate(top_idx):
                print(f"  {rank+1:2d}. {names[idx]:25s}  {imp_1d[idx]:.1f}")
        else:
            print(f"  (unexpected shape {imp_raw.shape}, skipping)")
    except Exception as e:
        print(f"  (not available: {e})")

    # ---- Save predictions plot ----
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('NGBoost + PharmHGT Features - AW_ST_CMC Prediction', fontsize=14)

        # Pred vs True with error bars (1 std)
        ax = axes[0]
        y_std = y_dist.std()
        ax.errorbar(y_test, y_pred, yerr=y_std, fmt='o', alpha=0.5,
                     capsize=2, markersize=4, ecolor='gray', markeredgecolor='k')
        lims = [min(y_test.min(), y_pred.min()) - 0.5, max(y_test.max(), y_pred.max()) + 0.5]
        ax.plot(lims, lims, 'r--', alpha=0.8, linewidth=1)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel('True AW_ST_CMC'); ax.set_ylabel('Predicted AW_ST_CMC')
        ax.set_title(f'Test R² = {test_r2:.4f}')
        ax.axis('square')

        # Residuals
        ax = axes[1]
        residuals = y_test - y_pred
        ax.scatter(y_pred, residuals, alpha=0.6, edgecolors='k', linewidth=0.5)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.8)
        ax.set_xlabel('Predicted AW_ST_CMC'); ax.set_ylabel('Residuals')
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
    print("SUMMARY — NGBoost + PharmHGT Features")
    print(f"{'='*60}")
    print(f"  Features:      {X_full.shape[1]}-dim")
    print(f"  Train:         {len(X_full)} (split {len(X_train)} train + {len(X_val)} val)")
    print(f"  Test:          {len(X_test)}")
    if study_best_value is not None:
        print(f"  Optuna:    {N_OPTUNA_TRIALS} trials, {N_FOLDS}-fold CV")
        print(f"  Best CV RMSE: {study_best_value:.4f}")
    else:
        print(f"  Params:    pretuned (skipped Optuna)")
    print(f"  Test RMSE:     {test_rmse:.4f}")
    print(f"  Test MAE:      {test_mae:.4f}")
    print(f"  Test R²:       {test_r2:.4f}")
    print(f"  Distribution:  Normal (probabilistic output)")

    # ---- 保存指标 & 更新索引 ----
    metrics = {
        'test_rmse': round(test_rmse, 4),
        'test_mae': round(test_mae, 4),
        'test_r2': round(test_r2, 4),
    }
    if study_best_value is not None:
        metrics['best_cv_rmse'] = round(study_best_value, 4)
    save_metrics(run_dir, metrics)
    update_index(run_dir, 'ngboost', metrics)


if __name__ == '__main__':
    main()
