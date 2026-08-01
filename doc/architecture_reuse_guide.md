# 分子性质预测项目架构复用指南

> 从 SurfPredict（表面活性剂性质预测）中提炼的可复用模式：**多目标 + 统一特征 + 多模型批量训练 + 自包含运行产物 + 免重训预测 API**。下一个分子性质项目可直接按此骨架搭建。

## 1. 总体分层

- **L1 数据层** — `data/surfpro/*.csv`（训练/测试/文献）：多目标，缺失率不同 → 每目标一个独立训练目录
- **L2 特征层** — `smiles_to_features_pharmhgt.py`：纯函数 `featurize(SMILES)` → (N, 522)；npy 缓存，MD5 为 key
- **L3 训练层** — `train/{target}/{model}_*.py`：每模型一个脚本，5 步固定骨架（特征 → 切分 → Optuna → 最终训练 → 落盘）
- **L4 运行/产物层** — `runs/{target}/{target}_{model}_{ts}/`：config.json + train.log + metrics.json + model.pkl + 双写 `_runs_index.csv`（target 级 + 全局）= 轻量实验追踪
- **L5 预测/服务层** — `use/use_models.py`：目录自动发现模型 → 自动选 best → 统一 `predict(SMILES)`

核心思想：**每一层都是自包含、可独立替换的**。换数据集只需改 L1+L2；加模型只需复制一个 L3 脚本；上线只需 L5。

## 2. 各层职责与关键约定

### L1 数据层：一个目标 = 一个训练目录
- 6 个目标缺失率不同（pCMC 9.8% ~ pC20 57.8%），按目标 `dropna` 后有效样本数不同。
- **约定：每个目标一套独立目录** `train/train_{TARGET}_models/`，目标名硬编码在脚本里。避免"换目标导致特征缓存失效"的坑（见 §4）。

### L2 特征层：特征工程与训练解耦
- 所有模型共用同一个 522 维特征向量；特征模块是**纯函数**，不依赖训练脚本。
- 缓存：`data/features/surfpro/{target}/` 下的 `.npy` + `metadata.json`，**cache key = MD5(SMILES 内容 + 目标列名)**。
- 提供两个入口：
  - `load_or_compute_features(train_csv, test_csv, target_col)` → 训练脚本用（批处理 + 缓存）
  - `smiles_to_features_pharmhgt(smiles)` → 预测 API 用（单分子）
- 先跑 `all_smiles_to_features.py` 预热缓存，所有训练脚本提速。

### L3 训练层：每模型一个脚本 + 固定 5 步骨架
```
setup_run() 建目录
  → load_or_compute_features() 取特征
  → train_test_split(0.125, seed=42) 切验证集
  → 调参：树模型=Optuna(TPE+MedianPruner+KFold) / 深度模型=固定架构
  → 全量训练 best params → 评估 → save_metrics + update_index
```
- **树模型 Optuna，深度模型固定架构**（MLP 可选用 Optuna）。XGBoost 额外加了 Top-K holdout 过滤，最精细。
- 深度模型把 522 维向量 reshape 成 `(batch, 522, 1)` 当序列喂 RNN/Transformer。

### L4 运行/产物层：一个运行目录 = 一个可复现单元
- 目录名约定承载元数据：`{target}_{model}_{timestamp}`（如 `pCMC_mlp_20260722_181638`）→ 无需数据库即可发现。
- 每次运行自动产出：`config.json`（全部超参+数据配置）、`train.log`（stdout tee）、`metrics.json`（test_rmse/mae/r2）、`model.pkl`、`pred_vs_true.png`、SHAP 图。
- `utils.py` 的 `setup_run / save_metrics / update_index` 把实验管理固化下来：**双写索引**（target 级追加 + 全局重写兼容旧格式）。

### L5 预测/服务层：免重训、免改代码上线
- 只依赖 **目录名约定 + config.json + 索引 CSV**，自动扫描 `runs/{target}/` 发现模型。
- `model_name='best'` → 按 `test_rmse` 最低自动选模型；也支持指定模型、`'all'` 集成平均、CLI。
- **双序列化约定（关键）**：
  - sklearn 系 → `joblib.dump` 直接存；
  - PyTorch 系 → 存 checkpoint dict：`{model_type, 全部超参, input_dim, state_dict}`。
  - 预测包在本地**重新搭网络**（`_MLPRegressor/_RNNRegressor/_TransformerRegressor`），无需 import 训练脚本。
- **y_scale 约定**：量级大的目标（Gamma_max）训练时缩放 1e6，缩放因子写进 `config.json`，预测时自动还原。

## 3. 五个最值得复用的决策

| 决策 | 为什么好 |
|------|---------|
| 目录名编码元数据 + 索引 CSV 双写 | 零成本的实验追踪，跨模型对比一眼可见 |
| config.json 记录一切 | 每个 run 自含复现信息，不依赖外部 tracker |
| 特征纯函数 + MD5 内容缓存 | 特征与训练解耦，缓存自动失效，绝不脏读 |
| Torch 存超参+state_dict 而非整个类 | 预测层可不 import 训练代码，部署无重依赖 |
| 目标硬编码 + 每目标独立目录 | 回避多目标样本集不同的缓存失效难题 |

## 4. 踩过的坑（换项目时提前规避）

1. **特征缓存与目标绑定**：`dropna` 随目标变化 → 缓存 (X,y) 对数量随之变化。**已解决**：每目标独立缓存目录；不要复用别的目标的缓存 X（AW_ST_CMC 会丢 129/843 样本）。
2. **目标硬编码的代价**：6 个目录是 `utils.py`/训练脚本的拷贝，改逻辑要同步 6 处。下个项目若目标量小，可改为"一个模板脚本 + 目标配置参数化"。
3. **预测包与训练脚本耦合**：曾因 import 训练脚本导致部署依赖爆炸。**已解决**：模型重建逻辑收敛到 `use_models.py` 内部。

## 5. 下一个项目复用清单（Copy → 替换）

1. `cp -r train/train_pCMC_models/ train/train_{TARGET}_models/`（全套骨架：utils.py + 训练脚本 + 特征模块）
2. 替换特征模块里的 `smiles_to_features_*` 为你的分子/对象特征，更新 `FEATURE_DIM`
3. 替换数据路径与 `TARGET_COL`（或参数化）
4. 按需增删 L3 模型脚本；深模型保持"存超参+state_dict"约定
5. 复制 `use/use_models.py`，改 `TARGETS` 集合即可得到同款预测 API
6. 跑 `all_smiles_to_features.py` 预热缓存 → 逐个训练脚本 → `use/use_models.py --list` 验证索引

**验收标准**：任一训练脚本产出完整 run 目录（config/log/metrics/model/plot）+ 索引更新；`SmilesPredict(SMILES)` 无需重训即可预测。
