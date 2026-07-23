from use_models import SmilesPredictor, quick_predict, list_models,SmilesPredict

# ---- 方式 1：自动选择最佳模型 ----
predictor = SmilesPredictor()
pred = predictor.predict('CCO')
print(f'Predicted pCMC: {pred:.4f}')

# ---- 方式 2：指定模型名称（自动加载最新版本） ----
predictor = SmilesPredictor(model_name='catboost')
predictor = SmilesPredictor(model_name='xgboost')
predictor = SmilesPredictor(model_name='mlp')

# ---- 方式 3：指定具体运行目录 ----
predictor = SmilesPredictor(run_dir='runs/xgboost_20260722_193605')

# ---- 方式 4：加载所有模型做集成预测 ----
predictor = SmilesPredictor(model_name='all')

# ---- 方式 5：快速单次预测（无需创建对象） ----
pred = quick_predict('CCO')
pred = quick_predict('CCO', model_name='catboost')

# ---- 查看可用模型 ----
df = list_models()
print(df)

# ---- 批量预测 ----
smiles_list = ['CCO', 'CCC(=O)O', 'c1ccccc1']
preds = predictor.predict(smiles_list)
for smi, p in zip(smiles_list, preds):
        print(f'{smi}: {p:.4f}')

# ---- 同时返回特征向量 ----
pred, features = predictor.predict('CCO', return_features=True)
print(features.shape)  # (522,)

# ---- 方式 0：一行预测（最简单） ----
pred = SmilesPredict('CCO')
pred = SmilesPredict('CCO', model_name='catboost')
pred = SmilesPredict(['CCO', 'CCC(=O)O'])          # 批量
pred, feats = SmilesPredict('CCO', return_features=True)  # +特征向量