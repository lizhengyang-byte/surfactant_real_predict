"""
experiment1.py — 实验一：容量匹配消融（必做）
===============================================

矩阵：6 个模型 × 6 个目标 = 36 组训练（per-target）。
  MLP-original   5层×488            ~1.0M   论文/文档原型，复跑
  MLP-small      3层×64             ~40K    新增：小容量非序列化反向对照
  RNN-original   3层×64             ~80K    论文/文档原型，复跑
  RNN-large      容量≈MLP-original   ~1.0M   新增：容量匹配后序列模型是否恢复
  Transformer-original 3层×d64      ~150K    论文/文档原型，复跑
  Transformer-large    容量≈MLP      ~1.0M   新增：同上

输出：results/exp1_capacity.csv（长表）、results/exp1_capacity.json、
      results/exp1_capacity_pivot.csv。
用法：python supp_experiments/experiment1.py [--resume]
"""
import os
import sys
import time
import json
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (set_seed, DEVICE, TARGET_NAMES, TARGET_CSV_COLS, YSCALE_GMAX,
                    RESULTS_DIR, load_prepared, get_target_data,
                    build_model_configs, count_params, train_one_model, evaluate)

LOG = os.path.join(RESULTS_DIR, "exp1_training.log")
RESULT_JSON = os.path.join(RESULTS_DIR, "exp1_capacity.json")
RESULT_CSV = os.path.join(RESULTS_DIR, "exp1_capacity.csv")
PIVOT_CSV = os.path.join(RESULTS_DIR, "exp1_capacity_pivot.csv")


def log_print(*args, **kwargs):
    line = " ".join(str(a) for a in args)
    print(line, **kwargs)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_done():
    """加载已完成结果（用于 --resume）。"""
    if os.path.isfile(RESULT_JSON):
        with open(RESULT_JSON, encoding="utf-8") as f:
            d = json.load(f)
        return set((m, t) for m in d.get("results", {}) for t in d["results"][m])
    return set()


def load_existing():
    """加载磁盘上已有的全部结果（JSON 结果字典 + CSV 行），供 --resume 回填使用。

    关键：--resume 时必须把既有结果载入 all_results/rows 的"起点"，
    否则每次 save_partial 全量覆写会把已完成的历史组合覆盖丢失。
    (2026-10-10 修复：第一次 resume 曾因未回灌导致 12 组结果被写掉。)
    """
    results, rows = {}, []
    if os.path.isfile(RESULT_JSON):
        with open(RESULT_JSON, encoding="utf-8") as f:
            d = json.load(f)
        results = d.get("results", {})
    if os.path.isfile(RESULT_CSV):
        try:
            rdf = pd.read_csv(RESULT_CSV)
            rows = [row.to_dict() for _, row in rdf.iterrows()]
        except Exception:
            rows = []
    return results, rows


def save_partial(capacity, all_results, rows, df):
    # 合并磁盘上既有 JSON，避免覆写扔掉已完成组合
    merged = {}
    if os.path.isfile(RESULT_JSON):
        try:
            with open(RESULT_JSON, encoding="utf-8") as f:
                merged = json.load(f).get("results", {})
        except Exception:
            merged = {}
    for m, td in all_results.items():
        merged.setdefault(m, {}).update(td)
    all_results = merged
    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump({"capacity_report": capacity, "results": all_results},
                  f, ensure_ascii=False, indent=2, default=str)
    if df is not None:
        df.to_csv(RESULT_CSV, index=False, encoding="utf-8-sig")
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="跳过已完成 (模型,目标) 组合")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        if not args.resume:
            f.write("\n=== 实验一：容量匹配消融 ===\n")
        f.write(f"device={DEVICE}, resume={args.resume}\n")

    set_seed(42)

    data = load_prepared()
    configs_full = build_model_configs()
    configs = {k: v for k, v in configs_full.items() if not k.startswith("_")}
    capacity = configs_full.get("_capacity_report")
    done = load_done() if args.resume else set()

    log_print("=" * 78)
    log_print("实验一：容量匹配消融（6 模型 × 6 目标 = 36 组 per-target 训练）")
    log_print(f"device: {DEVICE}")
    log_print("=" * 78)
    log_print("\n[参数量统计]")
    for name, cfg in configs.items():
        log_print(f"  {name:22s} params={cfg['params']:>10,}  ({cfg['desc']})")
    if capacity:
        log_print(f"\n  MLP/RNN-original 参数量比: {capacity['MLP/RNN-original_ratio']}x")
        log_print(f"  MLP/Transformer-original 参数量比: {capacity['MLP/Transformer-original_ratio']}x")
        log_print(f"  RNN-large 搜索结果: {json.dumps(capacity['RNN-large'], ensure_ascii=False)}")
        log_print(f"  Transformer-large 搜索结果: {json.dumps(capacity['Transformer-large'], ensure_ascii=False)}")

    rows = []
    all_results = {}
    if args.resume:
        all_results, rows = load_existing()
        log_print(f"[resume] 已载入既有结果 {sum(len(v) for v in all_results.values())} 组、CSV {len(rows)} 行")
    t0 = time.time()

    for j, (tname, tcol) in enumerate(zip(TARGET_NAMES, TARGET_CSV_COLS)):
        X_train, y_train, X_test, y_test = get_target_data(data, j)
        from sklearn.model_selection import train_test_split
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.125, random_state=42)
        unscale = YSCALE_GMAX if tname == "Gamma_max" else None
        log_print(f"\n[目标 {tname} ({tcol})] train={len(X_tr)} val={len(X_val)} test={len(X_test)}")

        for name, cfg in configs.items():
            if (name, tname) in done:
                log_print(f"  skip {name} × {tname} (已完成)")
                continue
            m = cfg["make"]()
            set_seed(42)
            log_print(f"  --- {name:22s} params={count_params(m):,} ---", end="")
            tt = time.time()
            model, best_epoch = train_one_model(m, X_tr, y_tr, X_val, y_val)
            res = evaluate(model, X_test, y_test, unscale=unscale)
            dt = time.time() - tt
            log_print(f"  epochs={best_epoch:3d}  R2={res['R2']:+.4f}  "
                      f"RMSE={res['RMSE']:.4g}  MAE={res['MAE']:.4g}  ({dt:.0f}s)")
            rows.append({
                "Model": name, "Target": tname, "Target_col": tcol,
                "params": cfg["params"],
                "R2": res["R2"], "RMSE": res["RMSE"], "MAE": res["MAE"],
                "best_epoch": best_epoch, "seconds": round(dt, 1),
            })
            all_results.setdefault(name, {})[tname] = res
            df_cur = pd.DataFrame(rows)
            save_partial(capacity, all_results, rows, df_cur)

    df = pd.DataFrame(rows)
    df.to_csv(RESULT_CSV, index=False, encoding="utf-8-sig")

    # ---- 汇总表 ----
    pivot = df.pivot(index="Model", columns="Target", values="R2")
    pivot["Mean"] = pivot.mean(axis=1)
    pivot = pivot.reindex(["MLP-original", "MLP-small", "RNN-original", "RNN-large",
                           "Transformer-original", "Transformer-large"])
    pivot.to_csv(os.path.join(RESULTS_DIR, "exp1_capacity_pivot.csv"), encoding="utf-8-sig")
    log_print("\n[R² 汇总表 (rows=模型, cols=目标)]")
    log_print(pivot.round(4).to_string())

    log_print("\n[关键比较]")
    for tname in TARGET_NAMES:

        def g(model):
            v = all_results[model][tname]["R2"] if model in all_results else np.nan
            return float(v)
        log_print(f"  {tname:10s} MLP-orig={g('MLP-original'):+.3f} "
                  f"MLP-small={g('MLP-small'):+.3f} "
                  f"RNN-orig={g('RNN-original'):+.3f} RNN-large={g('RNN-large'):+.3f} "
                  f"Trans-orig={g('Transformer-original'):+.3f} Trans-large={g('Transformer-large'):+.3f}")

    log_print(f"\n全部完成，耗时 {time.time()-t0:.0f}s → results/exp1_capacity.csv/.json")


if __name__ == "__main__":
    main()