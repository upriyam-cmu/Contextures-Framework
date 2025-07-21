import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, Literal, Union
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

from utils.types import DataFrame, ArrayLike

class KNNProbe:
    """
    KNN probe for downstream evaluation.
    Supports both classification and regression tasks.
    """
    def __init__(self, 
                 task_type: Literal["classification", "regression"] = "classification",
                 n_neighbors: int = 5,
                 **kwargs):
        self.task_type = task_type
        self.n_neighbors = n_neighbors
        if task_type == "classification":
            self.model = KNeighborsClassifier(n_neighbors=n_neighbors, **kwargs)
        else:
            self.model = KNeighborsRegressor(n_neighbors=n_neighbors, **kwargs)
        self._is_fitted = False

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self._is_fitted = True

    def predict(self, X):
        if not self._is_fitted:
            raise RuntimeError("KNN probe must be fitted before making predictions")
        return self.model.predict(X)

    def predict_proba(self, X):
        if self.task_type != "classification":
            raise ValueError("predict_proba is only available for classification tasks")
        if not self._is_fitted:
            raise RuntimeError("KNN probe must be fitted before making predictions")
        return self.model.predict_proba(X)

    def evaluate(self, X, y) -> Dict[str, float]:
        y_true = y
        y_pred = self.predict(X)
        metrics = {}
        if self.task_type == "classification":
            metrics["accuracy"] = accuracy_score(y_true, y_pred)
        else:
            metrics["r2_score"] = r2_score(y_true, y_pred)
            metrics["mse"] = mean_squared_error(y_true, y_pred)
            metrics["rmse"] = np.sqrt(metrics["mse"])
        return metrics


def train_knn_probe(
    features: Union[torch.Tensor, ArrayLike, DataFrame],
    targets: Union[torch.Tensor, ArrayLike, DataFrame],
    task_type: Literal["classification", "regression"] = "classification",
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    X_train=None, y_train=None, X_val=None, y_val=None, X_test=None, y_test=None,
    n_neighbors: int = 5,
    **probe_kwargs
) -> Tuple[KNNProbe, Dict[str, Any]]:
    """
    Train a KNN probe on extracted features.
    If X_train/y_train/X_val/y_val/X_test/y_test are provided, use them directly (no splitting).
    Otherwise, split features/targets internally.
    """
    # Convert to numpy for splitting
    if isinstance(features, torch.Tensor):
        features_np = features.cpu().numpy()
    elif isinstance(features, pd.DataFrame):
        features_np = features.values
    else:
        features_np = features
    
    if isinstance(targets, torch.Tensor):
        targets_np = targets.cpu().numpy()
    elif isinstance(targets, pd.DataFrame):
        targets_np = targets.values
    elif isinstance(targets, pd.Series):
        targets_np = targets.values
    else:
        targets_np = targets
    
    # Use provided splits if available
    if X_train is not None and y_train is not None and X_test is not None and y_test is not None:
        if X_val is not None and y_val is not None:
            pass
        else:
            X_val, y_val = None, None
    else:
        np.random.seed(random_state)
        n_samples = len(features_np)
        indices = np.random.permutation(n_samples)
        test_split = int(n_samples * (1 - test_size))
        val_split = int(test_split * (1 - val_size))
        train_indices = indices[:val_split]
        val_indices = indices[val_split:test_split]
        test_indices = indices[test_split:]
        X_train = features_np[train_indices]
        y_train = targets_np[train_indices]
        X_val = features_np[val_indices]
        y_val = targets_np[val_indices]
        X_test = features_np[test_indices]
        y_test = targets_np[test_indices]
    
    # Initialize and train probe
    probe = KNNProbe(task_type=task_type, n_neighbors=n_neighbors, **probe_kwargs)
    probe.fit(X_train, y_train)
    
    # Evaluate on test set
    test_metrics = probe.evaluate(X_test, y_test)
    
    # Compile results
    results = {
        "test_metrics": test_metrics,
        "data_splits": {
            "train_size": len(X_train),
            "val_size": len(X_val) if X_val is not None else 0,
            "test_size": len(X_test)
        }
    }
    
    return probe, results


def extract_features(encoder: torch.nn.Module, dataloader, device: str = "auto") -> Tuple[torch.Tensor, torch.Tensor]:
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    encoder.to(device)
    encoder.eval()
    features_list = []
    targets_list = []
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 2:
                x, y = batch
            else:
                x = batch[0]
                y = batch[1] if len(batch) > 1 else None
            x = x.to(device)
            features = encoder(x)
            features_list.append(features.cpu())
            if y is not None:
                targets_list.append(y.cpu())
    features = torch.cat(features_list, dim=0)
    targets = torch.cat(targets_list, dim=0) if targets_list else None
    return features, targets 
