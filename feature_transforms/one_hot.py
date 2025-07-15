# feature_transforms/one_hot.py

"""
Quick one-hot via pandas.get_dummies
"""

from __future__ import annotations

import pandas as pd

from utils.registry import register_transform
from ._base import BaseTransform

@register_transform('one_hot')
class OneHot(BaseTransform):
    def fit(self, X: pd.DataFrame) -> OneHot:
        self._cols_: list[str] = pd.get_dummies(X, dummy_na = False).columns.tolist()
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_t = pd.get_dummies(X, dummy_na = False)

        # align to training columns, unseen categories -> 0
        for col in self._cols_:
            if col not in X_t:
                X_t[col] = 0
                
        return X_t[self._cols_]