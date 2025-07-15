# feature_transforms/yeo_johnson.py

"""
Yeo-Johnson power transformation (Gaussianization)

Parameters: 
standardize - bool - if True, the output is shifted to mean = 0, var = 1
"""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import PowerTransformer

from utils.registry import register_transform
from ._base import BaseTransform

@register_transform('yeo_johnson')
class YeoJohnsonTransform(BaseTransform):
    def __init__(self, standardize: bool = True) -> None:
        self.standardize = standardize
        self._pt = PowerTransformer(method = 'yeo-johnson', standardize = standardize)
    
    def fit(self, X: pd.DataFrame) -> YeoJohnsonTransform:
        self._pt.fit(X)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_t = self._pt.transform(X)
        return pd.DataFrame(X_t, index = X.index, columns = X.columns)