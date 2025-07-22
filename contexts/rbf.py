import torch
from torch import Tensor
import numpy as np
from utils.registry import register_context
from utils.types import DataFrame
from contexts.base import Contexts

def rbf_kernel(x, x_prime, gamma):
    """
    Computes the RBF kernel between two sets of data points.
    Args:
        x (torch.Tensor): (n_samples_x, n_features)
        x_prime (torch.Tensor): (n_samples_x_prime, n_features)
        gamma (float): Bandwidth parameter.
    Returns:
        torch.Tensor: (n_samples_x, n_samples_x_prime)
    """
    diff = x.unsqueeze(1) - x_prime.unsqueeze(0)
    distances = torch.sum(diff ** 2, dim=2)
    return torch.exp(- gamma * distances)


def eigh(mat, device):
    if device == 'cpu':
        eigenvals, eigenvects = np.linalg.eigh(mat)
    else:
        try:
            kx_tensor = mat.clone().detach().float()
            eigenvals, eigenvects = torch.linalg.eigh(kx_tensor)
            eigenvals = eigenvals.cpu().numpy()
            eigenvects = eigenvects.cpu().numpy()
        except:
            eigenvals, eigenvects = np.linalg.eigh(mat)
    return eigenvals, eigenvects


@register_context('rbf')
class RBF(Contexts):
    def __init__(self, gamma: float = 1.0, max_rows: int = 1000, device: str = 'cpu', num_contexts: int = 1):
        self.gamma = gamma
        self.max_rows = max_rows
        self.device = device
        self.num_contexts = num_contexts

    def fit(self, dataset: DataFrame) -> None:
        X = torch.tensor(dataset.values, dtype=torch.float32, device=self.device) \
            if not torch.is_tensor(dataset) else dataset.to(self.device)
        if X.shape[0] > self.max_rows:
            idx = np.random.choice(X.shape[0], self.max_rows, replace=False)
            X = X[idx]
        X = X / (torch.norm(X, dim=1, keepdim=True) + 1e-8)
        self.X_train = X
        kx = rbf_kernel(self.X_train, self.X_train, self.gamma)
        self.mean_val = torch.mean(kx)
        eigenvals, eigenvects = eigh(kx.cpu().numpy(), self.device)
        self.largest_eigenval = eigenvals[-1]
        self.context_dim = self.X_train.shape[0]  # For kernel features, output dim = n_train

    def _sample(self, x: Tensor) -> Tensor:
        # x: (batch_size, num_features)
        k_feat = self._transform_single(x)  # (batch_size, n_train)
        k_feat_multi = k_feat.unsqueeze(1).expand(-1, self.num_contexts, -1)  # (batch_size, num_contexts, n_train)
        return k_feat_multi


