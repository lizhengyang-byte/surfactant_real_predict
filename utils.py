"""
utils.py — 实验运行管理与日志记录工具

每个训练脚本在 main() 开头调用 setup_run()，自动：
  1. 创建 runs/{model}_{timestamp}/ 目录
  2. 将 stdout 同时重定向到终端和 train.log
  3. 保存 config.json（超参数 + 数据配置）
  4. 训练结束时保存 metrics.json + 追加 runs/_runs_index.csv

用法：
    from utils import setup_run, save_metrics, update_index

    run_dir = setup_run('catboost', {'model': 'CatBoost', ...})
    # ... 训练代码（print 自动写入 train.log）...
    save_metrics(run_dir, {'test_rmse': 0.508, ...})
    update_index(run_dir, 'catboost', {'test_rmse': 0.508, ...})
"""

import os, sys, json, csv, atexit
from datetime import datetime

RUNS_DIR = 'runs'
INDEX_PATH = os.path.join(RUNS_DIR, '_runs_index.csv')


class _StreamTee:
    """将 stdout 同时写入文件和控制台。"""

    def __init__(self, file_path):
        self.log_file = open(file_path, 'w', encoding='utf-8')
        self.original = sys.stdout

    def write(self, data):
        self.log_file.write(data)
        self.original.write(data)

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
    run_name = f'{model_name}_{timestamp}'
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


def update_index(run_dir, model_name, metrics):
    """将本次运行结果追加到 runs/_runs_index.csv（方便横向对比）。"""
    os.makedirs(RUNS_DIR, exist_ok=True)

    row = {
        'model': model_name,
        'run_dir': os.path.relpath(run_dir),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    row.update(metrics)

    file_exists = os.path.isfile(INDEX_PATH)
    with open(INDEX_PATH, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"索引已更新: {INDEX_PATH}")
