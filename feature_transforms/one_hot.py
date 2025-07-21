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
        if X.shape[1] == 0:
            self._cols_ = []
            return self
        df_dummies = pd.get_dummies(X, dummy_na = False)
        self._cols_ = df_dummies.columns.tolist()
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if X.shape[1] == 0 or not getattr(self, '_cols_', None):
            return pd.DataFrame(index = X.index, columns = [])
        
        X_t = pd.get_dummies(X, dummy_na = False)

        # align to fit-time columns, filling unseen with zeros
        for col in self._cols_:
            if col not in X_t:
                X_t[col] = 0
        return X_t[self._cols_]