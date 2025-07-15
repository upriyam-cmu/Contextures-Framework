# feature_transforms/standardize.py

"""
Column-wise Z-score scaling: (x - mu) / sigma
"""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler

from utils.registry import register_transform
from ._base import BaseTransform

@register_transform('standardize')
class Standardize(BaseTransform):
    def __init__(self) -> None:
        self._scaler = StandardScaler()
    
    def fit(self, X: pd.DataFrame) -> Standardize:
        self._scaler.fit(X)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_t = self._scaler.transform(X)
        return pd.DataFrame(X_t, index = X.index, columns = X.columns)