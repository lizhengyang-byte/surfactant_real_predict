"""
all_smiles_to_features.py — 特征工程预运算：计算所有 522 维特征并缓存

运行方式（只需执行一次）：
    python all_smiles_to_features.py

作用：
    1. 从 data/surfpro/ 读取训练集和测试集 SMILES
    2. 计算全部 522 维 PharmHGT 风格分子特征
    3. 缓存至 data/features/surfpro/AW_ST_CMC（.npy + metadata.json）
    4. 后续训练脚本 train_*.py 直接从缓存加载，跳过重算

缓存失效策略：
    训练/测试集的 SMILES 列 MD5 哈希值记录在 metadata.json 中。
    若数据文件变更导致哈希不匹配，自动触发重算。
"""

from train.train_AW_ST_CMCmodels.smiles_to_features_pharmhgt import load_or_compute_features

if __name__ == '__main__':
    print("=" * 60)
    print("特征工程预运算：522-dim PharmHGT 风格特征")
    print("=" * 60)

    X_train, y_train, X_test, y_test = load_or_compute_features(
        force_recompute=True,
        verbose=True
    )

    print("\n" + "=" * 60)
    print("预运算完成！特征已缓存至 data/features/surfpro/AW_ST_CMC/")
    print("=" * 60)
    print(f"  训练集特征 (X_train): {X_train.shape}")
    print(f"  训练集目标 (y_train): {y_train.shape}")
    print(f"  测试集特征 (X_test):  {X_test.shape}")
    print(f"  测试集目标 (y_test):  {y_test.shape}")
    print()
    print("现在可以直接运行任意训练脚本（无需重复特征计算）：")
    print("  python train_catboost_use_pharmhgt_features.py")
    print("  python train_lightgbm_use_pharmhgt_features.py")
    print("  python train_xgboost_use_pharmhgt_features.py")
    print("  python train_mlp_use_pharmhgt_features.py")
    print("  python train_rnn_use_pharmhgt_features.py")
    print("  python train_transformer_use_pharmhgt_features.py")
