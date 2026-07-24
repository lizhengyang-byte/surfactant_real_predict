"""
shap_utils.py — SHAP 分析共用工具

提供：
 - setup_matplotlib()         配置 matplotlib 中文字体
 - load_features()           从缓存加载 522 维特征
 - find_latest_run()         从 runs/pC20/ 自动定位最新权重目录
 - get_top_features()        获取 top-N 重要特征索引
 - get_sample_indices()      获取最佳/最差/中位数预测样本索引
 - feature_name()            获取特征名称
"""
import os, re, warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

FEATURE_DIR = os.path.join('data', 'features', 'surfpro', 'pC20')
RUNS_DIR = os.path.join('runs', 'pC20')

# 惰性导入特征名称（模块级首次访问时加载）
_FEATURE_NAMES = None


def _get_feature_names():
    global _FEATURE_NAMES
    if _FEATURE_NAMES is None:
        from smiles_to_features_pharmhgt import FEATURE_NAMES as FN
        _FEATURE_NAMES = FN
    return _FEATURE_NAMES


def load_features():
    """从特征缓存加载数据。

    Returns:
        X_train: ndarray (n_train, 522)
        X_test:  ndarray (n_test, 522)
        y_train: ndarray (n_train,)
        y_test:  ndarray (n_test,)
    """
    X_train = np.load(os.path.join(FEATURE_DIR, 'X_train.npy'))
    y_train = np.load(os.path.join(FEATURE_DIR, 'y_train.npy'))
    X_test = np.load(os.path.join(FEATURE_DIR, 'X_test.npy'))
    y_test = np.load(os.path.join(FEATURE_DIR, 'y_test.npy'))
    return X_train, X_test, y_train, y_test


def find_latest_run(model_prefix):
    """在 runs/pC20/ 中查找该模型最新（按时间戳）的运行目录。

    Args:
        model_prefix: 模型前缀，如 'catboost'、'rf'、'mlp'。

    Returns:
        run_dir (str): 最新运行目录的路径。
    """
    pattern = re.compile(rf'^(?:pC20_)?{re.escape(model_prefix)}_(\d{{8}}_\d{{6}})$')
    candidates = []
    for d in os.listdir(RUNS_DIR):
        m = pattern.match(d)
        if m and os.path.isdir(os.path.join(RUNS_DIR, d)):
            from datetime import datetime
            ts = datetime.strptime(m.group(1), '%Y%m%d_%H%M%S')
            candidates.append((ts, d))
    if not candidates:
        raise FileNotFoundError(
            f'[shap_utils] 未找到前缀为 "{model_prefix}" 的运行目录。'
        )
    candidates.sort(key=lambda x: x[0], reverse=True)
    return os.path.join(RUNS_DIR, candidates[0][1])


def get_top_features(shap_values, n=5):
    """返回 mean |SHAP| 最高的 n 个特征索引（降序）。"""
    return np.argsort(np.abs(shap_values).mean(0))[-n:][::-1]


def get_sample_indices(y_true, y_pred):
    """返回最佳/中位数/最差预测的样本索引。

    Returns:
        (best_idx, mid_idx, worst_idx): 三个索引
    """
    errors = np.abs(y_true - y_pred)
    sorted_idx = np.argsort(errors)
    mid_pos = len(sorted_idx) // 2
    return sorted_idx[0], sorted_idx[mid_pos], sorted_idx[-1]


def feature_name(idx):
    """返回第 idx 个特征的名称（带括号索引，方便查询）。"""
    names = _get_feature_names()
    name = names[idx] if idx < len(names) else f'feat_{idx}'
    return f'{name} [{idx}]'


def is_large_ensemble(model, threshold=300):
    """检查模型是否为大型集成（树数量 > threshold），
    用于跳过耗时的交互矩阵计算。"""
    for attr in ['n_estimators', 'tree_count_', 'n_estimators_']:
        n = getattr(model, attr, None)
        if n is not None:
            return n > threshold
    # LightGBM
    if hasattr(model, 'num_trees'):
        return model.num_trees() > threshold
    # XGBoost
    if hasattr(model, 'get_params'):
        params = model.get_params()
        if 'n_estimators' in params:
            return params['n_estimators'] > threshold
    # 未知模型类型，保守返回 False
    return False


# ── matplotlib 中文字体配置 ──────────────────────────
_CN_FONT_SET = False


def _setup_cn_font():
    """全局配置 matplotlib 中文字体（仅执行一次）。"""
    global _CN_FONT_SET
    if _CN_FONT_SET:
        return
    _CN_FONT_SET = True

    # 尝试按优先级选择中文字体
    candidates = [
        'Microsoft YaHei',     # Windows 微软雅黑
        'SimHei',              # Windows 黑体
        'Noto Sans SC',        # Linux Noto
        'WenQuanYi Micro Hei', # Linux 文泉驿
        'PingFang SC',         # macOS 苹方
    ]
    chosen = None
    from matplotlib.font_manager import fontManager
    for name in candidates:
        matches = [f for f in fontManager.ttflist if f.name == name]
        if matches:
            chosen = name
            break
    if chosen is None and fontManager.ttflist:
        # 兜底：选任意一个可用字体
        chosen = fontManager.ttflist[0].name

    if chosen:
        plt.rcParams['font.family'] = chosen
        plt.rcParams['axes.unicode_minus'] = False
        import logging
        logging.getLogger('matplotlib').setLevel(logging.WARNING)


# 模块导入时自动配置
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    _setup_cn_font()
