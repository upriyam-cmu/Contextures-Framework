import numpy as np

import torch
from torch import nn

import xgboost as xgb

from tqdm.auto import tqdm

from abc import ABC, abstractmethod
from typing import Callable, Iterator, List, Literal, Optional, Tuple, Union, override
from typing_extensions import Self, Type
import hashlib
import json

from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import FeatureHasher
from sklearn.decomposition import PCA

from utils.registry import register_encoder


def _auto_diff_xgb_loss(loss_criterion, *args: np.ndarray, wrt: int = 0, device: str = 'cpu') -> Tuple[np.ndarray, np.ndarray]:
    device = torch.device(device)
    args_tens = [torch.tensor(arr, device=device) for arr in args]
    auto_diff_arg = args_tens[wrt].requires_grad_(True)

    loss, _ = loss_criterion(*args_tens)  # discard second ret val -- loss_dict

    grads = torch.autograd.grad(loss, auto_diff_arg, create_graph=True)[0]  # (N, D)
    hess_diag = sum([
        torch.abs(torch.autograd.grad(
            grads[:, idx],
            auto_diff_arg,
            grad_outputs=torch.ones_like(grads[:, idx]),
            retain_graph=True,
        )[0])
        for idx in range(grads.shape[-1])
    ])

    return grads.detach().cpu().numpy(), hess_diag.detach().cpu().numpy()

def _drop_trees(model: xgb.Booster, n_rounds_to_drop: int) -> xgb.Booster:
    # dump model state
    model_dump = model.save_raw('json')
    dump_obj = json.loads(model_dump.decode())

    # extract model info
    model_info = dump_obj['learner']['gradient_booster']['model']

    n_parallel_trees = model_info['gbtree_model_param']['num_parallel_tree']
    assert n_parallel_trees == '1', f"Can only process n_parallel_trees=1, got {n_parallel_trees!r}."

    n_trees_tot = int(model_info['gbtree_model_param']['num_trees'])
    itr_indptr, tree_info = model_info['iteration_indptr'], model_info['tree_info']

    n_rounds = len(itr_indptr) - 1
    assert n_trees_tot % n_rounds == 0, \
        f"Unexpected xgb dump data: n_rounds {n_rounds} does not divide into n_trees_total {n_trees_tot}"
    n_trees_per_round = n_trees_tot // n_rounds

    assert n_rounds > n_rounds_to_drop, \
        f"Cannot drop {n_rounds_to_drop} rounds from model with only {n_rounds} rounds. Must leave at least 1 round."

    # compute updated state
    n_trees_dropped = itr_indptr[n_rounds_to_drop]  # drop oldest trees
    new_num_trees = n_trees_tot - n_trees_dropped
    new_itr_indptr = [
        n_trees_at_round - n_trees_dropped
        for n_trees_at_round in itr_indptr[n_rounds_to_drop:]
    ]
    new_tree_info = tree_info[n_trees_dropped:]
    new_trees = [
        {**tree, 'id': new_id}
        for new_id, tree in enumerate(model_info['trees'][n_trees_dropped:])
    ]

    # save updated state
    model_info['gbtree_model_param']['num_trees'] = str(new_num_trees)
    model_info['iteration_indptr'] = new_itr_indptr
    model_info['tree_info'] = new_tree_info
    model_info['trees'] = new_trees

    # load updated model state
    new_model_state = json.dumps(dump_obj)
    model.load_model(bytearray(new_model_state, 'utf-8'))

    # return model
    return model

class XGBEncoderBase(ABC):
    seed: Optional[int]
    rng: np.random.Generator
    _torch_objective: Optional[nn.Module]

    @property
    def _custom_torch_objective(self):
        torch_obj = self._torch_objective
        if torch_obj is None: return None

        def xgb_objective(preds: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
            n, d = preds.shape
            A, B = preds, dtrain.get_label().reshape(n, d)

            return _auto_diff_xgb_loss(torch_obj, A, B, wrt=0, device='cpu')

        return xgb_objective

    def __init__(
        self,
        seed: Optional[int] = None,
        feature_mode: Literal['onehot', 'hashing', 'raw', 'pca'] = 'onehot',
        feature_dim: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()

        self._torch_objective = None

        if seed is not None:
            class_hash = int(hashlib.md5(self.__class__.__name__.encode()).hexdigest(), 16)
            self.seed = (seed + class_hash) & 0xFFFFFFFF
        else:
            self.seed = None
        self.set_rng()

        self.encoder, self.encoder_pred_leaf = self._build_encoder(
            feature_mode=feature_mode,
            feature_dim=feature_dim,
        )

    @staticmethod
    def _build_encoder(feature_mode: str, feature_dim: Optional[int]) -> Tuple[Callable[[np.ndarray], np.ndarray], bool]:
        if feature_mode == 'raw':
            pred_leaf = False
            encoder = lambda ft_mat: ft_mat

        elif feature_mode == 'onehot':
            pred_leaf = True

            one_hot = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoder = lambda leaf_mat: one_hot.fit_transform(leaf_mat)

        elif feature_mode == 'hashing':
            pred_leaf = True

            assert feature_dim is not None, "feature_dim must be specified for hashing mode"
            hasher = FeatureHasher(n_features=feature_dim, input_type='string')
            stringify = lambda leaf_mat: [[f"tree{j}_leaf{int(l)}" for j, l in enumerate(r)] for r in leaf_mat]

            encoder = lambda leaf_mat: hasher.transform(stringify(leaf_mat)).toarray()

        elif feature_mode == 'pca':
            pred_leaf = True

            assert feature_dim is not None, "feature_dim must be specified for PCA mode"
            one_hot = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            pca = PCA(n_components=feature_dim)

            encoder = lambda leaf_mat: pca.fit_transform(one_hot.fit_transform(leaf_mat))

        else:
            raise ValueError(f"Unknown feature_mode='{feature_mode}'")

        return encoder, pred_leaf

    def set_rng(self):
        self.rng = np.random.default_rng(seed=self.seed)

    def init(self, X: np.ndarray) -> Self:
        return self

    def train(self, torch_objective: nn.Module) -> Self:
        self._torch_objective = torch_objective
        return self

    @abstractmethod
    def fit_batch(self, X: np.ndarray, y: np.ndarray, *, num_rounds: Optional[int] = None, reset_model: bool = False) -> Self:
        pass

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        pass

    @abstractmethod
    def transform(self, X: np.ndarray, pred_leaf: Optional[bool] = None) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def exists(self) -> bool:
        """Check if model has been trained"""
        pass

    # torch nn.Module shims
    def to(self, *args, **kwargs) -> Self:
        return self

    def parameters(self):
        return []

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x_np = x.detach().cpu().numpy()
        out_raw = self.transform(x_np, pred_leaf=self.encoder_pred_leaf)
        out_np = self.encoder(out_raw)
        return torch.tensor(out_np)

@register_encoder('XGBoostEncoder')
class XGBoostEncoder(XGBEncoderBase):
    def __init__(
        self,
        objective: Union[str, callable, None] = None,
        num_rounds: int = 50,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        pred_leaf: bool = False,
        num_classes: Optional[int] = None,
        n_rounds_size_limit: Optional[int] = None,
        **kwargs,
    ):
        """
        objective: XGBoost target objective/criterion
        num_rounds: Number of boosting rounds/trees in model
        max_depth: Maximum tree depth of model
        learning_rate:
        pred_leaf: Whether to predict targets or indices of tree nodes
        num_classes: Number of classes (only applies to multi-class classification objectives)
        n_rounds_size_limit: If # of rounds > limit, prune oldest trees (per fit-batch)

        Example objectives:
          - 'reg:squarederror'
          - 'binary:logistic'
          - 'multi:softprob'
        """
        super().__init__(**kwargs)

        self.params = {
            'base_score': 0.5,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'verbosity': 0,
        }
        if isinstance(objective, str):
            self.params['objective'] = objective
            self.custom_objective = None
        else:
            self.custom_objective = objective

        if num_classes is not None:
            self.params['num_class'] = num_classes
        if self.seed is not None:
            self.params['seed'] = self.seed

        self.num_rounds = num_rounds
        self.pred_leaf = pred_leaf
        self.n_rounds_size_limit = n_rounds_size_limit

        self.model: Optional[xgb.Booster] = None

    @override
    def fit_batch(self, X: np.ndarray, y: np.ndarray, *, num_rounds: Optional[int] = None, reset_model: bool = False) -> Self:
        # check input dims
        if 'multi_strategy' in self.params:
            assert y.ndim == 2, f"XGBoost targets must be 2-dimensional for multi-target prediction"
        else:
            assert y.ndim == 1, f"XGBoost targets must be 1-dimensional for single-target prediction"
        assert y.shape[0] == X.shape[0], f"XGBoost targets must have same number of rows as input"

        # prune trees if necessary
        num_rounds = num_rounds or self.num_rounds
        if not reset_model and self.model is not None and self.n_rounds_size_limit is not None:
            if num_rounds >= self.n_rounds_size_limit:
                reset_model = True
            else:
                end_n_rounds = self.model.num_boosted_rounds() + num_rounds
                if end_n_rounds > self.n_rounds_size_limit:
                    self.model = _drop_trees(
                        model=self.model,
                        n_rounds_to_drop=(end_n_rounds - self.n_rounds_size_limit),
                    )

        # update model
        self.model = xgb.train(
            params=self.params,
            dtrain=xgb.DMatrix(X, label=y),
            num_boost_round=num_rounds,
            xgb_model=(self.model if not reset_model else None),
            obj=(self._custom_torch_objective or self.custom_objective),
        )

        # return self
        return self

    @override
    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        return self.fit_batch(X, y, reset_model=True)

    @override
    def transform(self, X: np.ndarray, pred_leaf: Optional[bool] = None) -> np.ndarray:
        if pred_leaf is None:
            pred_leaf = self.pred_leaf

        return self.model.predict(xgb.DMatrix(X), pred_leaf=pred_leaf)

    @override
    @property
    def exists(self) -> bool:
        """Check if model has been trained"""
        return self.model is not None

@register_encoder('MultiLabelXGBoostEncoder')
class MultiLabelXGBoostEncoder(XGBoostEncoder):
    def __init__(
        self,
        one_tree_per_target: bool = False,
        num_targets: Optional[int] = None,
        **kwargs,
    ):
        # pass all args through to XGBoost base-class
        super().__init__(**kwargs)

        # enable multi-label regression
        self.params['tree_method'] = 'hist'
        self.params['multi_strategy'] = 'one_output_per_tree' if one_tree_per_target else 'multi_output_tree'

        if num_targets is not None:
            self.params['num_target'] = num_targets

@register_encoder('XGBoostEnsembleEncoder')
class XGBoostEnsembleEncoder(XGBEncoderBase):
    def __init__(
            self,
            xgb_class: Type[XGBEncoderBase],
            /,
            max_batch_size: int = 1000,
            no_tqdm: bool = False,
            pred_mini_ensmbl_size: Optional[int] = None,
            **xgb_kwargs,
        ):
        super().__init__(**xgb_kwargs)

        self.xgb_class = xgb_class
        self.xgb_kwargs = xgb_kwargs
        self.max_batch_size = max_batch_size
        self.pred_mini_ensmbl_size = pred_mini_ensmbl_size
        self.no_tqdm = no_tqdm
        self.ensemble: Optional[List[XGBEncoderBase]] = None

    @override
    def init(self, X: np.ndarray) -> Self:
        dataset_size, _ = X.shape
        ensemble_size = (dataset_size - 1) // self.max_batch_size + 1
        self.ensemble = [
            self.xgb_class(**self.xgb_kwargs).init(X)
            for _ in range(ensemble_size)
        ]
        return self

    @override
    def train(self, torch_objective: nn.Module) -> Self:
        for model in self.ensemble: model.train(torch_objective)
        return super().train(torch_objective)

    def _ensemble_iter(self, n_data: int, *, label: str, randomize: bool) -> Iterator[Tuple[XGBEncoderBase, np.ndarray]]:
        if self.ensemble is None:
            raise ValueError("Ensemble not initialized. Call init() before fit().")

        ensemble_size = len(self.ensemble)
        data_perm = self.rng.permutation(n_data) if randomize else np.arange(n_data)
        data_chunks = np.array_split(data_perm, ensemble_size)

        for tree, chunk_idx in tqdm(
            zip(self.ensemble, data_chunks),
            total=ensemble_size,
            desc=f"XGBoost Ensemble: {label}",
            leave=False,
            disable=self.no_tqdm,
        ):
            yield tree, chunk_idx

    @override
    def fit_batch(self, X: np.ndarray, y: np.ndarray, *, num_rounds: Optional[int] = None, reset_model: bool = False) -> Self:
        for tree, chunk_idx in self._ensemble_iter(X.shape[0], label='fit_batch', randomize=True):
            tree.fit_batch(X[chunk_idx], y[chunk_idx], num_rounds=num_rounds, reset_model=reset_model)
        return self

    @override
    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        for tree, chunk_idx in self._ensemble_iter(X.shape[0], label='fit', randomize=True):
            tree.fit(X[chunk_idx], y[chunk_idx])
        return self

    @override
    def transform(self, X: np.ndarray, pred_leaf: Optional[bool] = None) -> np.ndarray:
        if self.pred_mini_ensmbl_size is None:
            return np.stack([
                tree.transform(X, pred_leaf=pred_leaf)
                for tree in tqdm(self.ensemble, desc='XGBoost Ensemble: transform', leave=False, disable=self.no_tqdm)
            ], axis=-1).mean(axis=-1)

        ret = np.concatenate([
            tree.transform(X[chunk_idx], pred_leaf=pred_leaf)
            for tree, chunk_idx in self._ensemble_iter(X.shape[0], label='transform (0)', randomize=False)
        ], axis=0)

        n_iter = max(0, self.pred_mini_ensmbl_size - 1)
        if n_iter == 0:
            return ret

        for idx in range(n_iter):
            for tree, chunk_idx in self._ensemble_iter(X.shape[0], label=f'transform ({idx + 1})', randomize=True):
                ret[chunk_idx] += tree.transform(X[chunk_idx], pred_leaf=pred_leaf)
        return ret / (n_iter + 1)

    @override
    @property
    def exists(self) -> bool:
        """Check if model has been trained"""
        return self.ensemble is not None and all(tree.exists for tree in self.ensemble)
