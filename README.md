# Contextures

Contextures is a modular Python package for context-aware data transformations, feature engineering, and pipeline orchestration. It is designed for extensibility, with a focus on composable feature transforms and a registry-based system for managing components.

---

## Directory Structure & Module Descriptions

### adapters/
- **__init__.py, kernel_to_multi.py, multi_to_joint.py**  
  *Currently empty; reserved for future adapters to convert between different context types (e.g., kernel, multi, joint contexts).*

### contexts/
- **__init__.py, base.py, joint_context.py, kernel_context.py, multi_conditional_context.py, single_conditional_context.py**  
  *Currently empty; reserved for future context class definitions and logic, such as base context interfaces, joint context composition, kernel-based contexts, and conditional context handling.*

### downstream/
- **__init__.py**  
  *Empty; reserved for downstream tasks or models.*

### encoders/
- **__init__.py, mlp.py**  
  *Empty; reserved for encoder implementations such as Multi-Layer Perceptrons (MLPs) or other feature encoders.*

### feature_transforms/

#### __init__.py
- Imports and exposes all feature transforms and the `ColumnPipeline` for easy access at the package level.

#### _base.py
- Defines the abstract base class for all transforms:
  ```python
  from abc import ABC, abstractmethod
  import pandas as pd

  class BaseTransform(ABC):
      def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
          return self.fit(X).transform(X)
      @abstractmethod
      def fit(self, X: pd.DataFrame) -> 'BaseTransform': ...
      @abstractmethod
      def transform(self, X: pd.DataFrame) -> pd.DataFrame: ...
  ```
  All custom transforms must inherit from `BaseTransform` and implement `fit` and `transform`.

#### impute.py
- Implements the `Imputer` transform using scikit-learn's `SimpleImputer`.
- Supports strategies: `mean`, `median`, `most_frequent`, `constant`.
- Registered as 'impute' in the registry.
- Example:
  ```python
  from feature_transforms.impute import Imputer
  imputer = Imputer(strategy='median')
  X_filled = imputer.fit_transform(X)
  ```

#### one_hot.py
- Implements the `OneHot` transform for one-hot encoding using pandas' `get_dummies`.
- Registered as 'one_hot' in the registry.
- Ensures output columns match those seen during fitting (unseen categories become 0).
- Example:
  ```python
  from feature_transforms.one_hot import OneHot
  encoder = OneHot()
  X_encoded = encoder.fit_transform(X)
  ```

#### standardize.py
- Implements the `Standardize` transform using scikit-learn's `StandardScaler` for z-score normalization.
- Registered as 'standardize' in the registry.
- Example:
  ```python
  from feature_transforms.standardize import Standardize
  scaler = Standardize()
  X_scaled = scaler.fit_transform(X)
  ```

#### whiten.py
- Implements the `Whitening` transform for ZCA whitening (decorrelates features, stabilizes with `eps`).
- Registered as 'whiten' in the registry.
- Example:
  ```python
  from feature_transforms.whiten import Whitening
  whitener = Whitening(eps=1e-5)
  X_white = whitener.fit_transform(X)
  ```

#### yeo_johnson.py
- Implements the `YeoJohnsonTransform` for Yeo-Johnson power transformation (optionally standardizes output).
- Registered as 'yeo_johnson' in the registry.
- Example:
  ```python
  from feature_transforms.yeo_johnson import YeoJohnsonTransform
  yj = YeoJohnsonTransform(standardize=True)
  X_yj = yj.fit_transform(X)
  ```

#### pipeline.py
- Implements `ColumnPipeline` for chaining transforms on numeric and categorical columns.
- Accepts transform specs as strings (for default params) or dicts (for custom params).
- Example:
  ```python
  from feature_transforms.pipeline import ColumnPipeline
  pipe = ColumnPipeline(
      numeric=['standardize', {'name': 'impute', 'strategy': 'median'}],
      categorical=[{'name': 'impute', 'strategy': 'most_frequent'}, 'one_hot']
  )
  X_out = pipe.fit_transform(X)
  ```
- Internally, uses the registry to instantiate transforms by name.

### losses/
- **__init__.py**  
  *Empty; reserved for loss function implementations.*

### mixing/
- **__init__.py**  
  *Empty; reserved for mixing strategies or modules.*

### orchestration/
- **__init__.py**  
  *Empty; reserved for orchestration logic.*

### utils/

#### __init__.py
- Exposes registry and type utilities for easy import.

#### registry.py
- Implements a decorator-based registry for transforms and other components.
- Key functions:
  - `register_transform(name)`: Decorator to register a transform class.
  - `get_transform(name)`: Retrieve a registered transform by name.
  - `list_transforms()`: List all registered transforms.
- Example:
  ```python
  from utils.registry import register_transform, get_transform, list_transforms

  @register_transform('custom')
  class CustomTransform(BaseTransform): ...
  t = get_transform('custom')()
  all_transforms = list_transforms()
  ```

#### types.py
- Provides type aliases for numpy and pandas objects:
  ```python
  ArrayLike  # np.ndarray
  DataFrame  # pd.DataFrame
  Series     # pd.Series
  ```

---

## Example: Building a Feature Pipeline

```python
from feature_transforms.pipeline import ColumnPipeline

pipe = ColumnPipeline(
    numeric=[
        'standardize',
        {'name': 'impute', 'strategy': 'median'},
        {'name': 'whiten', 'eps': 1e-4}
    ],
    categorical=[
        {'name': 'impute', 'strategy': 'most_frequent'},
        'one_hot'
    ]
)

X_out = pipe.fit_transform(X)
```

---

## Notes

- Most directories are currently stubs for future expansion.
- The core functionality is in `feature_transforms` and `utils`.
- All transforms inherit from `BaseTransform` and are registered for use in pipelines.
- The registry system allows for easy extension and dynamic lookup of transforms.

---
