# feature_transforms/pipeline.py

"""
Pipeline for transformations

Example Usage:
pipe = ColumnPipeline(numeric = ['yeo_johnson', 'standardize'],
                      categorical = ['one_hot', 'impute'])

df_train = pipe.fit_transform(df_train)
df_test = pipe.transform(df_test)
"""

from __future__ import annotations

import pandas as pd
from typing import Sequence, Callable

from utils.registry import get_transform

class ColumnPipeline:
    def __init__(self, numeric: Sequence[str] | None = None,
                 categorical: Sequence[str] | None = None) -> None:
        self.numeric_names = list(numeric) if numeric else []
        self.categorical_names = list(categorical) if categorical else []

    def _make_chain(self, names: Sequence[str]) -> list[Callable]:
        return [get_transform(n)() for n in names]
    
    def fit(self, df: pd.DataFrame) -> ColumnPipeline:
        self._num_cols = df.select_dtypes(exclude = ['object', 'category', 'string']).columns.tolist()
        self._cat_cols = [c for c in df.columns if c not in self._num_cols]

        self._num_chain = self._make_chain(self.numeric_names)
        self._cat_chain = self._make_chain(self.categorical_names)

        num_df = df[self._num_cols]
        for tr in self._num_chain:
            num_df = tr.fit_transform(num_df)
        
        cat_df = df[self._cat_cols]
        for tr in self._cat_chain:
            cat_df = tr.fit_transform(cat_df)
        
        self.output_dim = num_df.shape[1] + cat_df.shape[1]
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        num_df = df[self._num_cols]
        for tr in self._num_chain:
            num_df = tr.transform(num_df)
        
        cat_df = df[self._cat_cols]
        for tr in self._cat_chain:
            cat_df = tr.transform(cat_df)
        
        return pd.concat([num_df, cat_df], axis = 1)
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)