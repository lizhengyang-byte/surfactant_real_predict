"""
prepare_data.py — 为补充实验构建统一多目标数据集（实验一 / 实验四共用前提）
============================================================================

- 读取 data/surfpro/surfpro_train.csv 与 surfpro_test.csv；
- 用论文原管线的 build_feature_vector（522 维 PharmHGT 风格特征）为**全部**分子一次性计算特征；
- 目标列顺序遵循文档要求：[pCMC, gamma_CMC(AW_ST_CMC), Gamma_max, A_min(Area_min), Pi_CMC, pC20]；
  Gamma_max 列已乘以 YSCALE_GMAX (1e6)；
- 缺失标签位置填 NaN，并给出掩码矩阵；
- 输出到 supp_experiments/data/prepared_data.npz 并加缓存（重复运行直接命中）。

本脚本只读项目原有文件，不写入 data/features、runs/ 等原始路径。
用法：
  python supp_experiments/prepare_data.py
（不依赖 CWD，路径全部基于本文件位置解析）
"""
import os
import sys
import time
import json

import numpy as np
import pandas as pd

# ---- 路径解析（与 CWD 无关） ----
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUPP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SUPP_DIR, "data")
RESULTS_DIR = os.path.join(SUPP_DIR, "results")
FEAT_MODULE_DIR = os.path.join(ROOT, "train", "train_pCMC_models")

TRAIN_CSV = os.path.join(ROOT, "data", "surfpro", "surfpro_train.csv")
TEST_CSV = os.path.join(ROOT, "data", "surfpro", "surfpro_test.csv")

# 论文原管线特征缓存的 pCMC 子集（用于一致性校验）
PCMC_CACHE_DIR = os.path.join(ROOT, "data", "features", "surfpro", "pCMC")

# 文档要求的目标顺序：pCMC, gamma_CMC(→AW_ST_CMC), Gamma_max, A_min(→Area_min), Pi_CMC, pC20
TARGET_NAMES = ["pCMC", "gamma_CMC", "Gamma_max", "A_min", "Pi_CMC", "pC20"]
TARGET_CSV_COLS = ["pCMC", "AW_ST_CMC", "Gamma_max", "Area_min", "Pi_CMC", "pC20"]
YSCALE_GMAX = 1e6
FEATURE_DIM = 522

OUT_NPZ = os.path.join(DATA_DIR, "prepared_data.npz")
OUT_META = os.path.join(DATA_DIR, "prepared_data_meta.json")


# ---------------------------------------------------------------------------
# 特征计算（复用论文原管线）
# ---------------------------------------------------------------------------
def get_featurizer():
    """导入论文原管线特征提取模块（只读复用，不改写原文件）。"""
    if FEAT_MODULE_DIR not in sys.path:
        sys.path.insert(0, FEAT_MODULE_DIR)
    import smiles_to_features_pharmhgt as fmod
    return fmod


def featurize_all(df):
    """对 DataFrame 所有 SMILES 计算 522 维特征，返回 (X, valid_mask)。"""
    fmod = get_featurizer()
    feats = []
    valid = []
    bad = []
    t0 = time.time()
    for idx, smi in enumerate(df["SMILES"].values):
        fv = fmod.build_feature_vector(str(smi))
        if fv is None:
            valid.append(False)
            bad.append(idx)
            feats.append(None)
        else:
            if len(fv) != FEATURE_DIM:
                raise RuntimeError(f"row {idx}: feature dim {len(fv)} != {FEATURE_DIM}")
            valid.append(True)
            feats.append(fv)
    X = np.array([f for f in feats if f is not None], dtype=np.float32)
    return X, np.array(valid, dtype=bool), bad, time.time() - t0


def crosscheck_with_paper_cache(X_train, train_df):
    """与论文原管线的 pCMC 特征缓存逐元素比对，确认口径一致。"""
    if not os.path.isfile(os.path.join(PCMC_CACHE_DIR, "X_train.npy")):
        return "pCMC 缓存缺失，跳过交叉校验"
    X_paper = np.load(os.path.join(PCMC_CACHE_DIR, "X_train.npy"))  # (1204, 522)
    y_paper = np.load(os.path.join(PCMC_CACHE_DIR, "y_train.npy"))
    # 我的顺序 = train CSV 中 pCMC 非空的行，按行序排列（全部分子统一计算后按索引筛选）
    pcmc_valid = train_df["pCMC"].notna().values
    X_mine = X_train[pcmc_valid]
    y_mine = train_df["pCMC"].values[pcmc_valid]
    if X_mine.shape != X_paper.shape:
        return (f"形状不一致 mine{X_mine.shape} vs paper{X_paper.shape}")
    # 特征逐元素一致性（允许小数截断误差）
    max_abs_diff = np.abs(X_mine - X_paper).max()
    max_y_diff = float(np.abs(y_mine - y_paper).max())
    feats_ok = bool(np.allclose(X_mine, X_paper, atol=1e-4))
    y_ok = bool(np.allclose(y_mine, y_paper, atol=1e-7))
    return {
        "max_feat_abs_diff": float(max_abs_diff),
        "max_y_abs_diff": max_y_diff,
        "features_match": feats_ok,
        "labels_match": y_ok,
    }


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 已有缓存 → 直接命中
    if os.path.isfile(OUT_NPZ) and os.path.isfile(OUT_META):
        d = np.load(OUT_NPZ)
        print(f"[Cache HIT] {OUT_NPZ}")
        print(f"  X_train{d['X_train'].shape}  X_test{d['X_test'].shape}")
        print(f"  Y_train{d['Y_train'].shape}  Y_test{d['Y_test'].shape}")
        meta = json.load(open(OUT_META, encoding="utf-8"))
        print(f"  meta: {json.dumps(meta, ensure_ascii=False)}")
        return

    print(f"读取训练集: {TRAIN_CSV}")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    print(f"  train {train_df.shape}, test {test_df.shape}")

    # ---- 特征计算（全部分子一次） ----
    print("计算训练集特征 ...")
    X_train, tr_valid, tr_bad, t_tr = featurize_all(train_df)
    print(f"  训练集: {X_train.shape}, 耗时 {t_tr:.1f}s, 失败 {len(tr_bad)} 行")
    if len(tr_bad):
        print("  失败行索引:", tr_bad[:20])

    print("计算测试集特征 ...")
    X_test, te_valid, te_bad, t_te = featurize_all(test_df)
    print(f"  测试集: {X_test.shape}, 耗时 {t_te:.1f}s, 失败 {len(te_bad)} 行")

    # ---- 目标矩阵（缺失 → NaN），Gamma_max 乘 1e6 ----
    n_tr = X_train.shape[0]
    n_te = X_test.shape[0]
    Y_train = np.full((n_tr, len(TARGET_NAMES)), np.nan, dtype=np.float32)
    Y_test = np.full((n_te, len(TARGET_NAMES)), np.nan, dtype=np.float32)
    for j, col in enumerate(TARGET_CSV_COLS):
        col_vals_tr = train_df[col].values[tr_valid]
        col_vals_te = test_df[col].values[te_valid]
        if col == "Gamma_max":
            col_vals_tr = col_vals_tr * YSCALE_GMAX
            col_vals_te = col_vals_te * YSCALE_GMAX
        Y_train[:, j] = col_vals_tr
        Y_test[:, j] = col_vals_te

    y_train_mask = ~np.isnan(Y_train)
    y_test_mask = ~np.isnan(Y_test)

    # ---- 摘要 ----
    print("\n目标标签统计 (train / test 有效样本数):")
    for j, (name, col) in enumerate(zip(TARGET_NAMES, TARGET_CSV_COLS)):
        print(f"  {name:10s} ({col:10s}): train={int(y_train_mask[:, j].sum())}  "
              f"test={int(y_test_mask[:, j].sum())}  "
              f"y_tr_mean={np.nanmean(Y_train[:, j]):.4g}")

    # ---- 与论文原管线 pCMC 缓存交叉校验 ----
    ck = crosscheck_with_paper_cache(X_train, train_df)
    print("\n交叉校验 (fresh 特征 vs 论文 pCMC 缓存):")
    print(" ", json.dumps(ck, ensure_ascii=False, indent=2))

    # ---- 保存 ----
    np.savez_compressed(
        OUT_NPZ,
        X_train=X_train, X_test=X_test,
        Y_train=Y_train, Y_test=Y_test,
        y_train_mask=y_train_mask, y_test_mask=y_test_mask,
    )
    meta = {
        "feature_dim": FEATURE_DIM,
        "target_names": TARGET_NAMES,
        "target_csv_cols": TARGET_CSV_COLS,
        "yscale_gmax": YSCALE_GMAX,
        "n_train": n_tr, "n_test": n_te,
        "train_valid_mask_count": int(tr_valid.sum()),
        "test_valid_mask_count": int(te_valid.sum()),
        "train_bad_rows": tr_bad[:50],
        "test_bad_rows": te_bad[:50],
        "crosscheck_vs_pcmc_cache": ck,
        "featurization_time_s": {"train": t_tr, "test": t_te},
    }
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\n保存到 {OUT_NPZ}")
    print(f"元数据: {OUT_META}")


if __name__ == "__main__":
    main()