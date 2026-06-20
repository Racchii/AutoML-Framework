import pandas as pd
from sklearn.model_selection import train_test_split
from .preprocessor import Preprocessor
from .models import ModelFactory
from .evaluator import Evaluator
from .tuner import HyperparameterTuner
from typing import Dict, Any

class AutoMLPipeline:
    def __init__(self, task_type: str, target_column: str, model_names: list = None, use_tuning: bool = False, tuning_trials: int = 10):
        self.task_type = task_type
        self.target_column = target_column
        self.model_names = model_names or ['random_forest']
        self.use_tuning = use_tuning
        self.tuning_trials = tuning_trials
        self.preprocessor = Preprocessor()
        self.models = {}
        self.results = {}

    def fit(self, df: pd.DataFrame):
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataframe.")

        # Handle missing target rows
        df = df.dropna(subset=[self.target_column])

        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        # Splitting
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Preprocessing
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)
        
        tuner = HyperparameterTuner(task_type=self.task_type, n_trials=self.tuning_trials) if self.use_tuning else None

        for model_name in self.model_names:
            params = {}
            if self.use_tuning:
                params = tuner.tune(model_name, X_train_processed, y_train)
                
            model = ModelFactory.get_model(self.task_type, model_name, params)
            model.fit(X_train_processed, y_train)
            
            y_pred = model.predict(X_test_processed)
            y_prob = None
            if hasattr(model, 'predict_proba') and self.task_type == 'classification':
                y_prob = model.predict_proba(X_test_processed)

            # Evaluation
            if self.task_type == 'classification':
                metrics = Evaluator.evaluate_classification(y_test, y_pred, y_prob)
            else:
                metrics = Evaluator.evaluate_regression(y_test, y_pred)
                
            # Extract Feature Importances
            feature_names = self.preprocessor.get_feature_names()
            importances = {}
            if hasattr(model, 'feature_importances_'):
                imp = model.feature_importances_
                importances = {name: float(val) for name, val in zip(feature_names, imp)}
            elif hasattr(model, 'coef_'):
                imp = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
                importances = {name: float(abs(val)) for name, val in zip(feature_names, imp)}
                
            # Sort importances
            importances = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True)[:10]) # Top 10

            self.models[model_name] = model
            self.results[model_name] = {
                "metrics": metrics,
                "feature_importances": importances
            }

        return self.results
