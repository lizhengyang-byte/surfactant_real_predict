"""
SurfPredict — Model Prediction Package

提供统一的分子性质预测接口，自动从 runs/ 加载训练好的模型权重。

用法:
    from use.use_models import SmilesPredictor

    predictor = SmilesPredictor()              # 自动加载最佳模型
    pred = predictor.predict('CCO')            # 单分子预测
"""

from .use_models import SmilesPredictor, SmilesPredict, quick_predict, list_models

__all__ = ['SmilesPredictor', 'SmilesPredict', 'quick_predict', 'list_models']
