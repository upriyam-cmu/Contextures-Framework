from torch import Tensor
from typing import Tuple, Optional, Dict, Any
import torch
from utils.types import DataFrame

class Contexts():
    def fit(self, dataset: DataFrame) -> None:
        '''
        Some context needs to fit itself before use
        '''
        pass

    def _sample(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, num_features)
        Returns:
            a: Corresponding contexts (batch_size, num_contexts, num_features)
        """
        pass
    
    def get_collate_fn(self):
        """
        Returns:
            A callable function that takes a batch of data and returns a tuple
            (x, a), where x is the original input and a is the sampled context.
        """
        # a: (batch_size, num_context_examples, num_features)
        def collate_fn(x_batch):
            x_batch = torch.stack([b[0] for b in x_batch], dim = 0)
            return x_batch, self._sample(x_batch)
        return collate_fn