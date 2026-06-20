from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
import numpy as np
from typing import Dict, Any

class Evaluator:
    @staticmethod
    def evaluate_classification(y_true, y_pred, y_prob=None) -> Dict[str, float]:
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred, average='weighted')
        }
        if y_prob is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    # Binary classification
                    metrics["roc_auc"] = roc_auc_score(y_true, y_prob[:, 1])
                else:
                    # Multiclass classification
                    metrics["roc_auc"] = roc_auc_score(y_true, y_prob, multi_class='ovr')
            except Exception as e:
                pass # AUC might fail if probabilities are malformed
        return metrics

    @staticmethod
    def evaluate_regression(y_true, y_pred) -> Dict[str, float]:
        metrics = {
            "mse": mean_squared_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred)
        }
        return metrics
