"""
experiment4.py — 实验四：特征排列随机化（强烈建议）
====================================================

目的：验证「522 维特征序列的列顺序不携带物理信息」这一前提。

矩阵：2 个序列模型（RNN-original、Transformer-original）× 6 个目标
      × (1 基线 + 5 排列) = 72 组 per-target 训练。
Baseline 直接复用实验一 RNN-original / Transformer-original 的 per-target 结果
（同一套特征、同一训练协议），因此实际新增 5 × 2 × 6 = 60 组训练。

输出：
  results/exp4_permutation.json  全部原始结果
  results/exp4_summary.csv       每 (模型,目标) 的 baseline/mean(perm)/ΔR² 汇总
  results/exp4_report.txt        终端统计 + 结论判断
用法：python supp_experiments/experiment4.py [--permutations 5]
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (set_seed, DEVICE, TARGET_NAMES, TARGET_CSV_COLS, YSCALE_GMAX,
                    RESULTS_DIR, load_prepared, get_target_data, SEED,
                    train_one_model, evaluate, count_params,
                    _make_rnn_orig, _make_trans_orig)

LOG = os.path.join(RESULTS_DIR, "exp4_training.log")
MODELS = {
    "RNN-original": _make_rnn_orig,
    "Transformer-original": _make_trans_orig,
}


def log_print(*args, **kwargs):
    line = " ".join(str(a) for a in args)
    print(line, **kwargs)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_baseline_from_exp1():
    """从实验一结果读取 RNN-original / Transformer-original 的 per-target 基线。"""
    p = os.path.join(RESULTS_DIR, "exp1_capacity.json")
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    return d.get("results", {})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--permutations", type=int, default=5,
                        help="随机排列次数（文档建议 5）")
    parser.add_argument("--resume", action="store_true", help="跳过已完成 (模型,目标) 组合")
    args = parser.parse_args()
    n_perm = args.permutations

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        if not args.resume:
            f.write("\n=== 实验四：特征排列随机化 ===\n")
        f.write(f"device={DEVICE}, n_permutations={n_perm}, resume={args.resume}\n")

    set_seed(42)
    data = load_prepared()
    n_feat = data["X_train"].shape[1]

    # ---- 基线（优先复用实验一） ----
    baseline = load_baseline_from_exp1()
    use_reused = baseline is not None and all(
        m in baseline for m in MODELS)
    all_results = {"baseline": {}}
    if args.resume:
        pp = os.path.join(RESULTS_DIR, "exp4_permutation.json")
        if os.path.isfile(pp):
            old = json.load(open(pp, encoding="utf-8"))
            all_results = {k: v for k, v in old.items() if k != "baseline"}
            log_print(f"[resume] 已载入既有排列结果 {len(all_results)-1} 组")
    if use_reused:
        log_print(f"[复用实验一基线] RNN-original / Transformer-original 6×2 per-target 结果")
    else:
        log_print("实验一结果缺失或模型不全，将重新训练基线（12 组）")

    # 记录每个 (模型,目标) 的原始基线 R²；若复用，仅引用。
    baseline_r2 = {}
    for name, make in MODELS.items():
        for j, (tname, tcol) in enumerate(zip(TARGET_NAMES, TARGET_CSV_COLS)):
            unscale = YSCALE_GMAX if tname == "Gamma_max" else None
            b0 = baseline.get(name, {}).get(tname) if use_reused else None
            if use_reused and b0 is not None:
                baseline_r2[(name, tname)] = b0["R2"]
                all_results["baseline"][name] = baseline.get(name)
                continue
            # 需要训练基线
            X_train, y_train, X_test, y_test = get_target_data(data, j)
            from sklearn.model_selection import train_test_split
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train, test_size=0.125, random_state=42)
            m = make()
            set_seed(42)
            model, _ = train_one_model(m, X_tr, y_tr, X_val, y_val)
            res = evaluate(model, X_test, y_test, unscale=unscale)
            all_results["baseline"].setdefault(name, {})[tname] = res
            baseline_r2[(name, tname)] = res["R2"]
            log_print(f"[基线] {name} × {tname}: R2={res['R2']:+.4f}")
            with open(os.path.join(RESULTS_DIR, "exp4_permutation.json"), "w",
                      encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)

    log_print("\n" + "=" * 78)
    log_print(f"实验四：特征排列随机化（{n_perm} 次独立列置换 × 2 序列模型 × 6 目标）")
    log_print(f"特征维度: {n_feat}")
    log_print("=" * 78)

    # ---- 排列实验 ----
    t0 = time.time()
    for pi in range(n_perm):
        perm_seed = SEED + pi + 1
        rng = np.random.RandomState(perm_seed)
        perm_idx = rng.permutation(n_feat)
        key = f"perm_{pi + 1}"
        if key in all_results:
            log_print(f"\n--- Permutation {pi+1}/{n_perm} (seed={perm_seed}) [已存在，跳过] ---")
            continue
        all_results[key] = {}
        log_print(f"\n--- Permutation {pi+1}/{n_perm} (seed={perm_seed}) ---")
        for name, make in MODELS.items():
            for j, (tname, tcol) in enumerate(zip(TARGET_NAMES, TARGET_CSV_COLS)):
                X_train, y_train, X_test, y_test = get_target_data(data, j)
                X_tr_p = X_train[:, perm_idx]
                X_te_p = X_test[:, perm_idx]
                from sklearn.model_selection import train_test_split
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X_tr_p, y_train, test_size=0.125, random_state=42)
                m = make()
                set_seed(42)
                tt = time.time()
                model, best_epoch = train_one_model(m, X_tr, y_tr, X_val, y_val)
                unscale = YSCALE_GMAX if tname == "Gamma_max" else None
                res = evaluate(model, X_te_p, y_test, unscale=unscale)
                dt = time.time() - tt
                all_results[key].setdefault(name, {})[tname] = res
                log_print(f"  {name:20s} × {tname:10s} R2={res['R2']:+.4f}  ({dt:.0f}s, ep={best_epoch})")
                with open(os.path.join(RESULTS_DIR, "exp4_permutation.json"), "w",
                          encoding="utf-8") as f:
                    json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)

    with open(os.path.join(RESULTS_DIR, "exp4_permutation.json"), "w",
              encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)

    # ---- 汇总统计 ----
    log_print("\n" + "=" * 78)
    log_print("实验结果汇总")
    log_print("=" * 78)
    summary_rows = []
    for name in MODELS:
        log_print(f"\n--- {name} ---")
        rows = []
        for tname in TARGET_NAMES:
            b = baseline_r2[(name, tname)]
            perm_r2 = [all_results[f"perm_{pi+1}"][name][tname]["R2"]
                       for pi in range(n_perm)]
            mean_perm = float(np.mean(perm_r2))
            diff = mean_perm - b
            rows.append((tname, b, mean_perm, diff))
            summary_rows.append({
                "Model": name, "Target": tname,
                "baseline_R2": round(b, 4), "mean_perm_R2": round(mean_perm, 4),
                "delta_R2": round(diff, 4),
            })
        dfm = pd.DataFrame(rows, columns=["Target", "baseline", "mean_perm", "ΔR2"])
        log_print(dfm.round(4).to_string(index=False))

    # 总体统计（文档判据用「均值|ΔR²|」+ 论文建议以原始数值判断）
    df_all = pd.DataFrame(summary_rows)
    per_cell_delta = []
    for name in MODELS:
        for tname in TARGET_NAMES:
            b = baseline_r2[(name, tname)]
            for pi in range(n_perm):
                per_cell_delta.append(all_results[f"perm_{pi+1}"][name][tname]["R2"] - b)
    mean_abs_delta = float(np.mean(np.abs(per_cell_delta)))
    mean_delta = float(np.mean(per_cell_delta))
    max_abs_delta = float(np.max(np.abs(per_cell_delta)))
    std_delta = float(np.std(per_cell_delta))
    per_run = df_all.groupby(["Model"]).agg(
        mean_delta=("delta_R2", lambda s: float(np.mean(np.abs(s.to_numpy())))))
    log_print("\n" + "-" * 78)
    log_print(f"总体统计（{len(per_cell_delta)} 个 baseline vs perm 单组配对）:")
    log_print(f"  均值 ΔR² (mean of all per-cell deltas): {mean_delta:+.4f}")
    log_print(f"  均值 |ΔR²|: {mean_abs_delta:.4f}")
    log_print(f"  最大 |ΔR²|: {max_abs_delta:.4f}")
    log_print(f"  ΔR² 标准差: {std_delta:.4f}")

    # 文档判据
    log_print("\n文档判据（用均值 |ΔR²| 判别）:")
    if mean_abs_delta < 0.03:
        verdict = "A"
        msg = ("✓ 结论：排列后 R² 变化 < 0.03，维度顺序确实不携带物理信息；"
               "直接证明「无序描述符」前提成立。")
    elif mean_abs_delta < 0.10:
        verdict = "B"
        msg = ("~ 结论：排列后 R² 变化中等（0.03–0.10），维度顺序可能携带少量信息；"
               "需在论文中说明特征工程中存在微弱维度间结构。")
    else:
        verdict = "C"
        msg = ("✗ 结论：排列后 R² 变化 > 0.10，维度顺序有显著信息；"
               "需重新审视「无序描述符」前提。")
    log_print(f"  判定：场景 {verdict}")
    log_print(f"  {msg}")

    # per-model per-target 最坏情形（论文写法需要）
    worst = df_all.loc[df_all["delta_R2"].abs().idxmax()]
    log_print(f"  最敏感组合: {worst['Model']} × {worst['Target']}, ΔR²={worst['delta_R2']:+.4f}")

    df_all.to_csv(os.path.join(RESULTS_DIR, "exp4_summary.csv"),
                  index=False, encoding="utf-8-sig")
    summary = {
        "n_permutations": n_perm,
        "mean_delta_R2": mean_delta,
        "mean_abs_delta_R2": mean_abs_delta,
        "max_abs_delta_R2": max_abs_delta,
        "std_delta_R2": std_delta,
        "verdict": verdict,
    }
    with open(os.path.join(RESULTS_DIR, "exp4_verdict.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log_print(f"\n完成，耗时 {time.time()-t0:.0f}s → results/exp4_permutation.json / exp4_summary.csv")


if __name__ == "__main__":
    main()