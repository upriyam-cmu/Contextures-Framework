# feature_transforms/impute.py

"""
Strategy choices for imputation:
 - mean
 - median
 - most_frequent
 - constant
 """

from __future__ import annotations

import pandas as pd
from sklearn.impute import SimpleImputer as _SKSimpleImputer

from utils.registry import register_transform
from ._base import BaseTransform

@register_transform('impute')
class Imputer(BaseTransform):
    def __init__(self, strategy: str = 'mean', fill_value = None) -> None:
        self.strategy = strategy
        self.fill_value = fill_value
        self._imp = _SKSimpleImputer(strategy = strategy, fill_value = fill_value)
    
    def fit(self, X: pd.DataFrame) -> Imputer:
        self._imp.fit(X)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_t = self._imp.transform(X)
        return pd.DataFrame(X_t, index = X.index, columns = X.columns)