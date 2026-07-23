import sys, os
# 将项目根目录加入 sys.path（任何子目录都需要这一步）
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from use.use_models import SmilesPredictor, quick_predict, list_models,SmilesPredict

pred = SmilesPredict('CCCCCCCCCCCCOC(C)COC(C)COC(C)COP(=O)(O)[O-].[Na+]', model_name='catboost')
print(f'Predicted pCMC: {pred:.4f}')