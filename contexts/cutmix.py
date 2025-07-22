import torch
from torch import Tensor
import numpy as np
from utils.registry import register_context
from utils.types import DataFrame

@register_context('cutmix')
class Cutmix:
    def __init__(self, corruption_rate: float = 0.5, num_context_samples: int = 1, device: str = 'cpu'):
        self.corruption_rate = corruption_rate
        self.num_context_samples = num_context_samples
        self.device = device

    def fit(self, dataset: DataFrame) -> None:
        # No fitting needed for cutmix, but store feature count
        X = torch.tensor(dataset.values, dtype=torch.float32, device=self.device) \
            if not torch.is_tensor(dataset) else dataset.to(self.device)
        self.n_features = X.shape[1]
        self.n_samples = X.shape[0]

    def _transform_single(self, x: Tensor) -> Tensor:
        # x: (batch_size, n_features)
        batch_size, n_features = x.shape
        # Sample random indices for mixing
        idx = torch.randint(0, batch_size, (batch_size, n_features), device=x.device)
        x_rand = x[idx, torch.arange(n_features).unsqueeze(0).expand(batch_size, -1)]
        corruption_mask = torch.rand(batch_size, n_features, device=x.device) < self.corruption_rate
        x_cutmix = torch.where(corruption_mask, x_rand, x)
        return x_cutmix

    def _transform_multiple(self, x: Tensor) -> Tensor:
        # x: (batch_size, n_features)
        batch_size, n_features = x.shape
        r = self.num_context_samples
        # Sample random indices for all r contexts in parallel
        idx = torch.randint(0, batch_size, (batch_size, r, n_features), device=x.device)
        x_expanded = x.unsqueeze(1).expand(-1, r, -1)  # (batch_size, r, n_features)
        # Gather random values for each context
        x_rand = torch.gather(x_expanded, 0, idx)
        corruption_mask = torch.rand(batch_size, r, n_features, device=x.device) < self.corruption_rate
        x_cutmix = torch.where(corruption_mask, x_rand, x_expanded)
        return x_cutmix

    def get_collate_fn(self):
        if self.num_context_samples == 1:
            def collate_fn(x_batch):
                return x_batch, self._transform_single(x_batch)
        else:
            def collate_fn(x_batch):
                return x_batch, self._transform_multiple(x_batch)
        return collate_fn
