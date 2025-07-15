# feature_transforms/_base.py

"""
Making sure all transformations inherit 'fit', 'transform', and 'fit_transform'
"""

from __future__ import annotations

import pandas as pd
from abc import ABC, abstractmethod

class BaseTransform(ABC):
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)
    
    @abstractmethod
    def fit(self, X: pd.DataFrame) -> BaseTransform:
        ...
    
    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        ...