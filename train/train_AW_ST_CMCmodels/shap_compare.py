"""
shap_compare.py — 跨模型 SHAP 排名对比

加载各模型的 shap_values.npy，对比 Top-15 特征排名的异同。
输出一张多面板对比图 + 特征一致性评分表。
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Make project root importable when run directly
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from train.train_AW_ST_CMCmodels.shap_utils import RUNS_DIR, FEATURE_DIR, _get_feature_names

# 模型列表（按 Test R² 降序）
MODELS = [
    ('catboost',   'CatBoost',   0.8733),
    ('cif',        'CIF',        0.8751),
    ('xgboost',    'XGBoost',    0.8706),
    ('ngboost',    'NGBoost',    0.8605),
    ('lightgbm',   'LightGBM',   0.8602),
    ('histgb',     'HistGB',     0.8280),
    ('rf',         'RandomForest', 0.8128),
]
# 按 Test R² 降序排列
MODELS.sort(key=lambda x: x[2], reverse=True)

N_TOP = 15
FEATURE_NAMES = _get_feature_names()


def load_shap(model_prefix):
    """加载模型的 shap_values.npy，返回 (shap_values, run_dir)。"""
    from train.train_AW_ST_CMCmodels.shap_utils import find_latest_run
    try:
        run_dir = find_latest_run(model_prefix)
    except FileNotFoundError:
        return None, None
    path = os.path.join(run_dir, 'shap_values.npy')
    if not os.path.exists(path):
        print(f'[WARN] {model_prefix}: shap_values.npy 不存在，跳过')
        return None, None
    sv = np.load(path)
    return sv, run_dir


def compute_rank_table():
    """为所有模型计算 Top-15 特征排名。

    Returns:
        rank_matrix: dict[model_name] -> dict[feat_idx] -> rank (1-based)
        mean_abs:    dict[model_name] -> (feat_indices, mean_abs_values)
    """
    rank_data = {}
    mean_abs_data = {}
    for mname, mlabel, _ in MODELS:
        sv, run_dir = load_shap(mname)
        if sv is None:
            continue
        # 每个特征的 mean |SHAP|
        ma = np.abs(sv).mean(0)
        # 按重要性降序排列的特征索引
        sorted_idx = np.argsort(ma)[::-1]
        # 排名映射
        ranks = {idx: rank + 1 for rank, idx in enumerate(sorted_idx)}
        rank_data[mname] = ranks
        mean_abs_data[mname] = (sorted_idx, ma)
        print(f'  {mlabel}: shap_values 已加载 ({sv.shape})')
    return rank_data, mean_abs_data


def plot_ranking_comparison(rank_data, mean_abs_data):
    """生成跨模型 Top-15 特征排名对比图。"""
    # 收集所有模型都有的特征 → 取所有模型 Top-N 的并集
    all_top_feats = set()
    for mname in rank_data:
        sorted_idx, _ = mean_abs_data[mname]
        all_top_feats.update(sorted_idx[:N_TOP])
    all_top_feats = sorted(all_top_feats)
    n_feats = len(all_top_feats)

    # 构建排名矩阵 (n_models × n_feats)，缺失填 NaN
    model_names = []
    model_labels = []
    rank_matrix = np.full((len(rank_data), n_feats), np.nan)
    for i, (mname, mlabel, r2) in enumerate(MODELS):
        if mname not in rank_data:
            continue
        model_names.append(mname)
        model_labels.append(f'{mlabel}\n(R²={r2})')
        ranks = rank_data[mname]
        for j, feat_idx in enumerate(all_top_feats):
            rank_matrix[len(model_names) - 1, j] = ranks.get(feat_idx, np.nan)

    # 特征名称（截断到 30 字符）
    feat_labels = []
    for idx in all_top_feats:
        name = FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else f'feat_{idx}'
        if len(name) > 30:
            name = name[:27] + '...'
        feat_labels.append(f'{name} [{idx}]')

    fig_height = max(6, n_feats * 0.35)
    fig, ax = plt.subplots(figsize=(14, fig_height))

    cmap = plt.cm.RdYlGn_r  # 红=排名低, 绿=排名高
    norm = mcolors.Normalize(vmin=1, vmax=N_TOP)
    im = ax.imshow(rank_matrix, aspect='auto', cmap=cmap, norm=norm)

    # 标注排名数字
    for i in range(len(model_names)):
        for j in range(n_feats):
            val = rank_matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val < N_TOP / 2 else 'black'
                ax.text(j, i, f'{int(val)}', ha='center', va='center',
                        fontsize=7, color=color)

    ax.set_xticks(range(n_feats))
    ax.set_xticklabels(feat_labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_labels, fontsize=9)
    ax.set_title('跨模型 Top-15 特征 SHAP 排名对比\n(绿色=高排名, 红色=低排名, 白色=未进前15)',
                 fontsize=12, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label('SHAP 排名', fontsize=9)

    plt.tight_layout()
    out_path = os.path.join('doc', 'report', 'shap_cross_model_ranking.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f'\n[SHAP] 跨模型排名对比图: {out_path}')


def plot_feature_agreement(rank_data, mean_abs_data):
    """绘制特征重要性一致性得分的条形图。"""
    # 一致性：每个模型 Top-15 中，被其他模型也列为 Top-N 的平均比例
    n_models = len(rank_data)
    agreement = {}
    for mname in rank_data:
        sorted_idx, _ = mean_abs_data[mname]
        top_set = set(sorted_idx[:N_TOP])
        other_counts = []
        for other_name in rank_data:
            if other_name == mname:
                continue
            other_sorted, _ = mean_abs_data[other_name]
            other_top = set(other_sorted[:N_TOP])
            overlap = len(top_set & other_top)
            other_counts.append(overlap / N_TOP)
        agreement[mname] = np.mean(other_counts) if other_counts else 0

    # 按一致性排序
    sorted_models = sorted(agreement.items(), key=lambda x: x[1], reverse=True)
    names = []
    scores = []
    colors = []
    for mname, score in sorted_models:
        for mn, ml, r2 in MODELS:
            if mn == mname:
                names.append(f'{ml}\n({r2})')
                break
        scores.append(score)
        colors.append('#2ecc71' if score > 0.5 else '#f39c12')

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(range(len(names)), scores, color=colors, height=0.6)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('与其它模型 Top-15 的平均重叠比例', fontsize=10)
    ax.set_title('各模型 SHAP 特征排名一致性\n(越高表示该模型特征偏好越具代表性)',
                 fontsize=11, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.invert_yaxis()

    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{score:.1%}', va='center', fontsize=9)

    plt.tight_layout()
    out_path = os.path.join('doc', 'report', 'shap_feature_agreement.png')
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f'[SHAP] 特征一致性图: {out_path}')


def print_consensus_features(rank_data, mean_abs_data):
    """输出所有模型一致认可的关键特征。"""
    # 计算每个特征在所有模型中的平均排名
    feat_ranks = {}  # feat_idx -> list of ranks
    for mname in rank_data:
        sorted_idx, _ = mean_abs_data[mname]
        for rank, idx in enumerate(sorted_idx[:N_TOP], 1):
            if idx not in feat_ranks:
                feat_ranks[idx] = []
            feat_ranks[idx].append(rank)

    print(f'\n{"="*60}')
    print('所有模型一致认可的关键特征 (均进入 Top-15)')
    print(f'{"="*60}')
    print(f'{"特征":<25} {"出现模型数":>10} {"平均排名":>10}')
    print('-' * 50)
    consensus = [(idx, ranks) for idx, ranks in feat_ranks.items()
                 if len(ranks) == len(rank_data)]
    consensus.sort(key=lambda x: sum(x[1]) / len(x[1]))
    for idx, ranks in consensus:
        name = FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else f'feat_{idx}'
        avg_rank = sum(ranks) / len(ranks)
        print(f'{name:<25} {len(ranks):>10} {avg_rank:>8.1f}')


def main():
    print('[SHAP] 加载各模型 SHAP 值...')
    rank_data, mean_abs_data = compute_rank_table()

    if len(rank_data) < 2:
        print('[SHAP] 至少需要 2 个模型的 SHAP 值才能对比')
        sys.exit(1)

    plot_ranking_comparison(rank_data, mean_abs_data)
    plot_feature_agreement(rank_data, mean_abs_data)
    print_consensus_features(rank_data, mean_abs_data)

    print(f'\n[SHAP] 跨模型对比完成，输出目录: doc/report/')


if __name__ == '__main__':
    main()
