from typing import Tuple, Optional, Dict, Any
import torch
from torch import Tensor
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.registry import register_context
from utils.types import DataFrame
from contexts.base import Contexts

@register_context('scarf')
class SCARF(Contexts):
    """
    SCARF (Self-supervised Contrastive Learning using Random Feature Corruption) knowledge component.
    
    Implements feature corruption for self-supervised learning by randomly replacing
    features with values sampled from marginal distributions.
    """
    def __init__(self, num_context_samples: int, distribution: str = 'uniform', corruption_rate: float = 0.6):
        self.num_context_samples = num_context_samples
        self.distribution = distribution
        self.corruption_rate = corruption_rate
        self.uniform_eps = 1e-6
    
    def fit(self, dataset: DataFrame) -> None:
        """
        Fit the SCARF knowledge to the dataset by computing feature bounds.
        
        Args:
            dataset: DataFrame containing the training data
        """
        self.context_dim = dataset.shape[1]
        
        # Compute feature bounds from the dataset
        features_low = dataset.min().values
        features_high = dataset.max().values
        
        # Store feature bounds
        self.features_low = torch.Tensor(features_low)
        self.features_high = torch.Tensor(features_high)
        
        # Initialize marginal distributions based on configuration
        if self.distribution == 'uniform':
            self.marginals = Uniform(
                self.features_low - self.uniform_eps, 
                self.features_high + self.uniform_eps
            )
        elif self.distribution == 'gaussian':
            mean = (self.features_high + self.features_low) / 2
            std = (self.features_high - self.features_low) / 4
            self.marginals = Normal(mean, std)
        elif self.distribution == 'bimodal':
            std = (self.features_high - self.features_low) / 8
            self.marginals_low = Normal(self.features_low, std)
            self.marginals_high = Normal(self.features_high, std)
        else:
            raise NotImplementedError(f"Unsupported prior distribution: {self.distribution}")
<<<<<<< HEAD
=======
    
    def get_collate_fn(self):
        if self.num_context_samples == 1:
            # a: (batch_size, num_features)
            def collate_fn(x_batch):
                return x_batch, self._transform_single(x_batch)
        else:
            # a: (batch_size, num_context_samples, num_features)
            def collate_fn(x_batch):
                return x_batch, self._transform_multiple(x_batch)
        return collate_fn
>>>>>>> origin/main

    def _sample(self, x: Tensor) -> Tensor:
        """
        Apply SCARF corruption to input features.
        Args:
            x: Input tensor of shape (batch_size, num_features)
        Returns:
            corrupted_x: Tensor of shape (batch_size, num_context_samples, num_features)
        """
        batch_size, num_features = x.size()
        r = self.num_context_samples
        # Create corruption mask for all r contexts in parallel
        corruption_mask = (torch.rand(batch_size, r, num_features, device=x.device) < self.corruption_rate)
        # Sample random values for all r contexts in parallel
        if self.distribution in ['uniform', 'gaussian']:
            # marginals.sample((batch_size, r)) returns (batch_size, r, num_features)
            x_random = self.marginals.sample(torch.Size((batch_size, r))).to(x.device)
        elif self.distribution == 'bimodal':
            x_random_low = self.marginals_low.sample(torch.Size((batch_size, r))).to(x.device)
            x_random_high = self.marginals_high.sample(torch.Size((batch_size, r))).to(x.device)
            mode_choice = torch.rand(batch_size, r, device=x.device) > 0.5
            mode_choice = mode_choice.unsqueeze(-1).expand(-1, -1, num_features)
            x_random = torch.where(mode_choice, x_random_low, x_random_high)
        # Expand x to (batch_size, r, num_features)
        x_expanded = x.unsqueeze(1).expand(-1, r, -1)
        # Apply corruption mask
        x_corrupted = torch.where(corruption_mask, x_random, x_expanded)
        return x_corrupted
