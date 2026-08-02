"""
utils.py — 实验运行管理与日志记录工具

每个训练脚本在 main() 开头调用 setup_run()，自动：
  1. 创建 runs/Gamma_max/pCMC_{model}_{timestamp}/ 目录
  2. 将 stdout 同时重定向到终端和 train.log
  3. 保存 config.json（超参数 + 数据配置）
  4. 训练结束时：保存 metrics.json + 双写索引
      ├─ runs/Gamma_max/_runs_index.csv   (target 专属)
      └─ runs/_runs_index.csv        (全局总览)

用法：
    from utils import setup_run, save_metrics, update_index

    run_dir = setup_run('catboost', {'model': 'CatBoost', ...})
    # ... 训练代码（print 自动写入 train.log）...
    save_metrics(run_dir, {'test_rmse': 0.508, ...})
    update_index(run_dir, 'catboost', {'test_rmse': 0.508, ...})
"""

import os, sys, json, csv, atexit
from datetime import datetime

TARGET_NAME = 'Gamma_max'
RUNS_DIR = os.path.join('runs', TARGET_NAME)
INDEX_PATH = os.path.join(RUNS_DIR, '_runs_index.csv')
GLOBAL_INDEX_PATH = os.path.join('runs', '_runs_index.csv')


class _StreamTee:
    """将 stdout 同时写入文件和控制台。"""

    def __init__(self, file_path):
        self.log_file = open(file_path, 'w', encoding='utf-8')
        self.original = sys.stdout

    def write(self, data):
        self.log_file.write(data)
        try:
            self.original.write(data)
        except UnicodeEncodeError:
            # 控制台编码（如 GBK）无法表示的字符用替代符写出，避免运行崩溃
            enc = getattr(self.original, 'encoding', None) or 'utf-8'
            self.original.write(data.encode(enc, errors='replace').decode(enc))

    def flush(self):
        self.log_file.flush()
        self.original.flush()

    def close(self):
        self.log_file.close()


def _restore_stdout():
    """恢复原始 stdout（atexit 注册，确保崩溃时也恢复）。"""
    if isinstance(sys.stdout, _StreamTee):
        sys.stdout.close()
    sys.stdout = sys.__stdout__


atexit.register(_restore_stdout)


def setup_run(model_name, config):
    """创建带时间戳的运行目录，重定向 stdout。

    Args:
        model_name: 模型标识，如 'catboost'、'xgboost'、'mlp'。
        config: 超参数配置字典 → 保存为 config.json。

    Returns:
        run_dir (str): 运行目录路径。
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f'{TARGET_NAME}_{model_name}_{timestamp}'
    run_dir = os.path.join(RUNS_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # 保存配置
    with open(os.path.join(run_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # 重定向 stdout
    sys.stdout = _StreamTee(os.path.join(run_dir, 'train.log'))

    print("=" * 60)
    print(f"{model_name} — 运行开始  {timestamp}")
    print(f"  输出目录:  {run_dir}")
    print("=" * 60)

    return run_dir


def save_metrics(run_dir, metrics):
    """保存评估指标到 run_dir/metrics.json。"""
    path = os.path.join(run_dir, 'metrics.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\n指标已保存: {path}")
    return path


def _write_index_row(filepath, fieldnames, row):
    """向单个索引文件追加一行（不存在时自动写入表头）。"""
    dir_path = os.path.dirname(filepath)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    file_exists = os.path.isfile(filepath)
    with open(filepath, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _rewrite_index_row(filepath, new_row, all_fieldnames):
    """重写索引文件：合并新旧列，保留旧行数据，追加新行。

    适用于全局索引：旧文件可能有不同的列结构，追加模式会导致列错位。
    """
    dir_path = os.path.dirname(filepath)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    # 读取旧行（如果存在）
    old_rows = []
    old_columns = []
    if os.path.isfile(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                old_columns = reader.fieldnames or []
                for r in reader:
                    # 过滤 None 键（兼容旧文件中列数不匹配的行）
                    clean = {k: v for k, v in r.items() if k is not None}
                    old_rows.append(clean)
        except (StopIteration, csv.Error):
            pass

    # 合并列：保持旧列顺序，新列追加末尾
    merged = list(dict.fromkeys(old_columns + all_fieldnames))

    # 重写全部
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=merged)
        writer.writeheader()
        for r in old_rows:
            writer.writerow(r)
        writer.writerow(new_row)


def update_index(run_dir, model_name, metrics):
    """将本次运行结果同时写入 target 专有索引和全局索引。

    双写:
      runs/Gamma_max/_runs_index.csv  — pCMC 专属索引（追加）
      runs/_runs_index.csv       — 全局总览索引（重写，兼容旧格式）
    """
    row = {
        'target': TARGET_NAME,
        'model': model_name,
        'run_dir': os.path.relpath(run_dir),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    row.update(metrics)

    # 1. 写入 target 专有索引（追加模式，列结构一致）
    fieldnames = list(row.keys())
    _write_index_row(INDEX_PATH, fieldnames, row)
    print(f"索引已更新: {INDEX_PATH}")

    # 2. 写入全局索引（重写模式，兼容旧文件列结构）
    _rewrite_index_row(GLOBAL_INDEX_PATH, row, fieldnames)
    print(f"索引已更新: {GLOBAL_INDEX_PATH}")
