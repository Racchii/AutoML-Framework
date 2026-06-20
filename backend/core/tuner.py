import optuna
from sklearn.model_selection import cross_val_score
import pandas as pd
from typing import Dict, Any, Callable
from .models import ModelFactory

class HyperparameterTuner:
    def __init__(self, task_type: str, n_trials: int = 10):
        self.task_type = task_type
        self.n_trials = n_trials
        # Suppress optuna logging for cleaner output
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    def tune(self, model_name: str, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Runs Optuna optimization and returns best parameters."""
        
        def objective(trial):
            params = self._get_search_space(trial, model_name)
            model = ModelFactory.get_model(self.task_type, model_name, params)
            
            scoring = 'accuracy' if self.task_type == 'classification' else 'neg_mean_squared_error'
            
            # Using 3-fold CV for speed
            scores = cross_val_score(model, X, y, cv=3, scoring=scoring)
            
            return scores.mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials)
        
        return study.best_params

    def _get_search_space(self, trial, model_name: str) -> Dict[str, Any]:
        if model_name == 'random_forest':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            }
        elif model_name == 'xgboost':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            }
        elif model_name in ['logistic_regression', 'linear_regression']:
            # Minimal tuning for simple linear models
            return {}
        else:
            return {}
