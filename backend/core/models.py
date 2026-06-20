from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
import xgboost as xgb
from typing import Dict, Any, Tuple

class ModelFactory:
    @staticmethod
    def get_model(task_type: str, model_name: str, params: Dict[str, Any] = None):
        params = params or {}
        
        models = {
            'classification': {
                'random_forest': RandomForestClassifier(**params),
                'logistic_regression': LogisticRegression(max_iter=1000, **params),
                'xgboost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', **params),
            },
            'regression': {
                'random_forest': RandomForestRegressor(**params),
                'linear_regression': LinearRegression(**params),
                'xgboost': xgb.XGBRegressor(**params),
            }
        }
        
        if task_type not in models:
            raise ValueError(f"Task type {task_type} not supported.")
            
        if model_name not in models[task_type]:
            raise ValueError(f"Model {model_name} not supported for task {task_type}.")
            
        return models[task_type][model_name]
