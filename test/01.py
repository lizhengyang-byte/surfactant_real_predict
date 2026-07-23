import sys, os
# 将项目根目录加入 sys.path（任何子目录都需要这一步）
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from use.use_models import SmilesPredictor, quick_predict, list_models,SmilesPredict

# pred = SmilesPredict('CCO', model_name='catboost')
# print(f'Predicted pCMC: {pred:.4f}')

file_path = 'data\\surfpro\\surfpro_test.csv'
import pandas as pd
df = pd.read_csv(file_path)
smiles_list = df['SMILES'].tolist()
pcmc_list = df['pCMC'].tolist()

# 创建预测器对象
predictor = SmilesPredictor(model_name='mlp')  # 可以选择不同的模型，如 'catboost', 'xgboost', 'mlp'

def test_predictor(predictor, smiles_list, pcmc_list):
    preds = predictor.predict(smiles_list)
    for smi, p, true in zip(smiles_list, preds, pcmc_list):
        print(f'{smi}: Predicted pCMC: {p:.4f}, True pCMC: {true:.4f}')
    return preds


# 计算R2
from sklearn.metrics import r2_score
preds = test_predictor(predictor, smiles_list, pcmc_list)
r2 = r2_score(pcmc_list, preds)
print(f'R2 score: {r2:.4f}')
