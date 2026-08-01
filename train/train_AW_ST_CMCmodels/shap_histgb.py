"""
shap_histgb.py — SHAP 全面分析：HistGB

自动加载 runs/ 下最新的 HistGB 权重和特征缓存，
输出 SHAP 特征重要性图到 run 目录。
"""
import os, joblib, warnings, traceback
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from train.train_AW_ST_CMCmodels.shap_utils import (
    load_features, find_latest_run,
    get_top_features, get_sample_indices,
    feature_name, is_large_ensemble
)

MODEL_NAME = 'histgb'
MODEL_LABEL = 'HistGB'


def main():
    run_dir = find_latest_run(MODEL_NAME)
    print(f'[SHAP] 模型: {MODEL_LABEL} ({MODEL_NAME})')
    print(f'[SHAP] 加载权重: {run_dir}')

    # ---- 1. 加载特征与模型 ----
    X_train, X_test, y_train, y_test = load_features()
    print(f'[SHAP] 特征: X_train {X_train.shape}, X_test {X_test.shape}')

    model = joblib.load(os.path.join(run_dir, 'model.pkl'))
    print(f'[SHAP] 模型加载完成')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
    y_pred = model.predict(X_test)

    # 保存 SHAP 值矩阵
    np.save(os.path.join(run_dir, 'shap_values.npy'), shap_values)

    # ---- 2. 蜂群图（全局特征重要性 + 方向性） ----
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(os.path.join(run_dir, 'shap_summary.png'),
                bbox_inches='tight', dpi=150)
    plt.close()
    print(f'[SHAP] 完成: shap_summary.png')

    # ---- 3. 平均 |SHAP| 条形图（Top-20） ----
    shap.summary_plot(shap_values, X_test, plot_type='bar',
                      show=False, max_display=20)
    plt.savefig(os.path.join(run_dir, 'shap_bar.png'),
                bbox_inches='tight', dpi=150)
    plt.close()
    print(f'[SHAP] 完成: shap_bar.png')

    # ---- 4. 依赖图（Top-5 特征） ----
    top_idx = get_top_features(shap_values, n=5)
    for rank, idx in enumerate(top_idx, 1):
        fname = feature_name(idx)
        shap.dependence_plot(
            idx, shap_values, X_test,
            feature_names=None,
            interaction_index='auto',
            show=False
        )
        safe_name = fname.replace('[', '_').replace(']', '')
        path = os.path.join(run_dir, f'shap_dependence_top{rank}_{safe_name}.png')
        plt.savefig(path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f'[SHAP] 完成: 依赖图 Top-{rank} {fname}')

    # ---- 5. 瀑布图（最佳/中位数/最差预测） ----
    try:
        explanation = explainer(X_test)  # shap.Explanation 对象
        best_idx, mid_idx, worst_idx = get_sample_indices(y_test, y_pred)

        samples = [
            (best_idx,  'best',   f'最佳预测 (|误差|={abs(y_test[best_idx]-y_pred[best_idx]):.3f})'),
            (mid_idx,   'median', f'中位数预测 (|误差|={abs(y_test[mid_idx]-y_pred[mid_idx]):.3f})'),
            (worst_idx, 'worst',  f'最差预测 (|误差|={abs(y_test[worst_idx]-y_pred[worst_idx]):.3f})'),
        ]
        for idx, tag, desc in samples:
            shap.plots.waterfall(explanation[idx], max_display=15, show=False)
            plt.gcf().suptitle(f'{MODEL_LABEL} — {desc}', fontsize=11)
            path = os.path.join(run_dir, f'shap_waterfall_{tag}.png')
            plt.savefig(path, bbox_inches='tight', dpi=150)
            plt.close()
            print(f'[SHAP] 完成: 瀑布图 {tag} ({desc})')
    except Exception as e:
        print(f'[SHAP] 瀑布图跳过 ({e})')

    # ---- 6. SHAP 交互作用热力图（Top-8 特征） ----
    if is_large_ensemble(model):
        print(f'[SHAP] 交互热力图跳过（大型集成，{MODEL_LABEL} 树数超过阈值）')
    else:
        try:
            top8_idx = get_top_features(shap_values, n=8)
            X_test_top8 = X_test[:, top8_idx]
            shap_interaction = explainer.shap_interaction_values(X_test)
            interaction_sub = shap_interaction[:, top8_idx, :][:, :, top8_idx]
            shap.summary_plot(
                interaction_sub, X_test_top8,
                feature_names=[feature_name(i) for i in top8_idx],
                plot_type='heatmap', show=False
            )
            plt.savefig(os.path.join(run_dir, 'shap_interaction_heatmap.png'),
                        bbox_inches='tight', dpi=150)
            plt.close()
            print(f'[SHAP] 完成: shap_interaction_heatmap.png')
        except Exception as e:
            print(f'[SHAP] 交互热力图跳过 ({e})')

    print(f'[SHAP] 全部完成，输出目录: {run_dir}')


if __name__ == '__main__':
    main()
