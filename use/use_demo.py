from use_models import SmilesPredictor, quick_predict, list_models,SmilesPredict



pred = SmilesPredict('CCO', model_name='catboost', target='logP')
pred = SmilesPredict(['CCO', 'CCC(=O)O'], model_name='catboost', target='logP')          # 批量
pred, feats = SmilesPredict('CCO', model_name='catboost', target='logP', return_features=True)  # +特征向量