# contexts/multi_conditional_context.py

"""
Multi-conditional context for SCARF (Self-supervised Contrastive Learning using Random Feature Corruption) augmentation.
"""

from typing import Tuple, Optional, Dict, Any
import torch
from torch import Tensor
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from utils.registry import register_transform
from utils.types import DataFrame


class BaseKnowledge(ABC):
    """Base class for knowledge components that can fit to data and transform it."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
    
    @abstractmethod
    def fit(self, dataset: DataFrame) -> None:
        """Fit the knowledge component to the dataset."""
        pass
    
    @abstractmethod
    def transform(self, x: Tensor, y: Optional[Tensor] = None, **kwargs) -> Tuple[Tensor, Optional[Tensor]]:
        """Transform input data using the fitted knowledge."""
        pass


class SCARFKnowledge(BaseKnowledge):
    """
    SCARF (Self-supervised Contrastive Learning using Random Feature Corruption) knowledge component.
    
    Implements feature corruption for self-supervised learning by randomly replacing
    features with values sampled from marginal distributions.
    """
    
    def fit(self, dataset: DataFrame) -> None:
        """
        Fit the SCARF knowledge to the dataset by computing feature bounds.
        
        Args:
            dataset: DataFrame containing the training data
        """
        # Compute feature bounds from the dataset
        features_low = dataset.min().values
        features_high = dataset.max().values
        
        # Get configuration parameters
        self.distribution = self.config.get('distribution', 'uniform')
        self.corruption_rate = self.config.get('corruption_rate', 0.5)
        self.uniform_eps = self.config.get('uniform_eps', 1e-6)
        
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

    def transform(self, x: Tensor, y: Optional[Tensor] = None, **kwargs) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Apply SCARF corruption to input features.
        
        Args:
            x: Input tensor of shape (batch_size, num_features)
            y: Optional target tensor
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (corrupted_x, y) where corrupted_x has the same shape as x
        """
        batch_size, _ = x.size()
        
        # Create corruption mask - corruption_rate of entries will be 1
        corruption_mask = torch.rand_like(x, device=x.device) < self.corruption_rate
        
        # Sample random values based on the distribution
        if self.distribution in ['uniform', 'gaussian']:
            x_random = self.marginals.sample(torch.Size((batch_size,))).to(x.device)
        elif self.distribution == 'bimodal':
            x_random_low = self.marginals_low.sample(torch.Size((batch_size,))).to(x.device)
            x_random_high = self.marginals_high.sample(torch.Size((batch_size,))).to(x.device)
            # Randomly choose between low and high modes
            mode_choice = torch.rand(batch_size, device=x.device) > 0.5
            x_random = torch.where(mode_choice.unsqueeze(1), x_random_low, x_random_high)
        
        # Apply corruption: replace features where corruption_mask is True
        x_corrupted = torch.where(corruption_mask, x_random, x)
        
        return x_corrupted, y


class MultiConditionalContext:
    """
    Multi-conditional context that generates multiple augmented versions of input data.
    
    This context applies SCARF augmentation to generate r different corrupted versions
    of the input features for self-supervised learning.
    """
    
    def __init__(self, knowledge: BaseKnowledge, num_augmentations: int = 1):
        """
        Initialize the multi-conditional context.
        
        Args:
            knowledge: Knowledge component (e.g., SCARFKnowledge) for augmentation
            num_augmentations: Number of augmented versions to generate (r)
        """
        self.knowledge = knowledge
        self.num_augmentations = num_augmentations
        self._is_fitted = False
    
    def fit(self, dataset: DataFrame) -> 'MultiConditionalContext':
        """
        Fit the context to the dataset.
        
        Args:
            dataset: Training dataset
            
        Returns:
            Self for chaining
        """
        self.knowledge.fit(dataset)
        self._is_fitted = True
        return self
    
    def transform(self, x: Tensor, y: Optional[Tensor] = None, **kwargs) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Generate multiple augmented versions of the input.
        
        Args:
            x: Input tensor of shape (batch_size, num_features)
            y: Optional target tensor
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (augmented_x, y) where augmented_x has shape (batch_size, num_augmentations, num_features)
        """
        if not self._is_fitted:
            raise RuntimeError("Context must be fitted before transformation")
        
        batch_size, num_features = x.size()
        
        # Generate multiple augmented versions
        augmented_versions = []
        for _ in range(self.num_augmentations):
            x_aug, _ = self.knowledge.transform(x, y, **kwargs)
            augmented_versions.append(x_aug)
        
        # Stack augmentations along a new dimension
        # Shape: (batch_size, num_augmentations, num_features)
        augmented_x = torch.stack(augmented_versions, dim=1)
        
        return augmented_x, y
    
    def transform_single(self, x: Tensor, y: Optional[Tensor] = None, **kwargs) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Generate a single augmented version of the input.
        
        Args:
            x: Input tensor of shape (batch_size, num_features)
            y: Optional target tensor
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (augmented_x, y) where augmented_x has the same shape as x
        """
        if not self._is_fitted:
            raise RuntimeError("Context must be fitted before transformation")
        
        return self.knowledge.transform(x, y, **kwargs)


# Registry registration for easy access
@register_transform('scarf')
class SCARFTransform:
    """
    SCARF transform that can be used in the feature pipeline.
    
    This is a wrapper around MultiConditionalContext for integration with the existing
    transform registry system.
    """
    
    def __init__(self, 
                 distribution: str = 'uniform',
                 corruption_rate: float = 0.5,
                 uniform_eps: float = 1e-6,
                 num_augmentations: int = 1):
        """
        Initialize SCARF transform.
        
        Args:
            distribution: Distribution type ('uniform', 'gaussian', 'bimodal')
            corruption_rate: Rate of feature corruption (0.0 to 1.0)
            uniform_eps: Epsilon for uniform distribution bounds
            num_augmentations: Number of augmented versions to generate
        """
        config = {
            'distribution': distribution,
            'corruption_rate': corruption_rate,
            'uniform_eps': uniform_eps
        }
        
        knowledge = SCARFKnowledge(config)
        self.context = MultiConditionalContext(knowledge, num_augmentations)
        self._is_fitted = False
    
    def fit(self, X: DataFrame) -> 'SCARFTransform':
        """Fit the SCARF transform to the data."""
        self.context.fit(X)
        self._is_fitted = True
        return self
    
    def transform(self, X: DataFrame) -> DataFrame:
        """
        Transform DataFrame to tensor, apply SCARF, and return as DataFrame.
        
        Note: This returns the original DataFrame as SCARF is primarily designed
        for tensor-based operations in self-supervised learning contexts.
        """
        if not self._is_fitted:
            raise RuntimeError("SCARF transform must be fitted before transformation")
        
        # For DataFrame compatibility, return original data
        # SCARF is typically used with tensors in self-supervised learning
        return X
    
    def transform_tensor(self, x: Tensor, y: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Transform tensor data using SCARF augmentation.
        
        Args:
            x: Input tensor
            y: Optional target tensor
            
        Returns:
            Tuple of (augmented_x, y)
        """
        if not self._is_fitted:
            raise RuntimeError("SCARF transform must be fitted before transformation")
        
        return self.context.transform(x, y)
