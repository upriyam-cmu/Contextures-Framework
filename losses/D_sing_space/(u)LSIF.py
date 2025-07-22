import torch
from torch import nn
from typing import Sequence, Union, List, Literal
from torch.nn import functional as F
from utils.registry import register_loss

@register_loss('uLSIF')
class uLSIF(nn.Module):
    """
    uLSIF is basically LoRA with weight decay (use AdamW) and normalization penalty.
    LoRA (Low-Rank Approximation) loss implementation.
    Naive version:
    L = -2 E_{x,a ~ P(x,a)} [\Phi(x)^T \Psi(a)] + E_{x ~ p(x), a ~ p(a)} [ (\Phi(x)^T \Psi(a) )^2],
    where \Phi(x) and \Psi(a) are the embeddings of inputs x and contexts a, respectively.
    
    Exponential parameterization version (inner product):
    L = -2 E_{x,a ~ P(x,a)} [\exp(\Phi(x)^T \Psi(a) / T)] + E_{x ~ p(x), a ~ p(a)} [ (\exp(\Phi(x)^T \Psi(a) / T))^2].

    
    Exponential parameterization version (squared):
    L = -2 E_{x,a ~ P(x,a)} [\exp(|| \Phi(x) - \Psi(a) ||^2 / T)] + E_{x ~ p(x), a ~ p(a)} [ || \Phi(x) - \Psi(a) ||^2  / T))^2].

    """
    def __init__(self, 
                 x_proj: nn.Module = None, 
                 a_proj: nn.Module = None,
                 exp_parameterization: Literal["inner_product", "squared"] = None,
                 temparature: float = 1.0,
                 ) -> None:
        """
        Initialize the LoRA loss module.
        Args:
        - x_proj: a MLP module that further projects inputs x to embeddings. \Phi'(x) = x_proj(\Phi(x))
        - a_proj: a MLP module that further projects contexts a to embeddings. \Psi'(a) = a_proj(\Psi(a))
        - exp_paramerization:  whether to use exponential parameterization. 
        - temparature: float, temperature for exp_parameterization, default is 1.0.
        """
        super(SVDLoRA, self).__init__()
        self.x_proj = x_proj
        self.a_proj = a_proj
        self.exp_parameterization = exp_parameterization
        self.temparature = temparature

    def forward(self, x: torch.Tensor, a: torch.Tensor,
                reduction: Literal["mean", "none"] = "mean",
                ) -> torch.Tensor:
        """
        Inputs:
        - x: embedding of inputs x, torch tensor of shape (N, D)
        - a: embedding of contexts a, torch tensor of shape (N,D) or (N, r, D), representing single context or r contexts for each input 
        
        Outputs:
        - lora_loss: LoRA loss, torch tensor of shape (N,) or scalar if mean reduction is applied.
        - loss_dict: Dictionary containing verbose information like the loss of the positive pairs and negative pairs.
        """
        if x_proj is not None:
            x = self.x_proj(x)
        if a_proj is not None:
            if a.ndim == 2:
                a = self.a_proj(a)
            elif a.ndim == 3:
                N, r, D = a.shape
                a = self.a_proj(a.view(N*r, D)).view(N, r, -1)
        N, D = x.shape  # Batch size and embedding dimension

        if a.ndim == 2:
            assert x.shape == a.shape, "Input tensors must have the same shape"
            
            # Dot products between embeddings
            if self.exp_parameterization is None:
                dot_products = x @ a.T # (N, N)
            elif self.exp_parameterization == "inner_product":
                dot_products = torch.exp(x @ a.T / self.temparature) # (N, N)
            elif self.exp_parameterization == "squared":
                # Z[i, j] = exp(|| X[i] - A[j] ||^2 / T)
                squared_diff = ((x[:, None, :] - a[None, :, :]) ** 2).sum(dim=-1)
                dot_products = torch.exp(squared_diff / self.temperature) # (N, N)
            
            # Compute the dot products for positive pairs
            sim_pos = dot_products.diag()  # Similarity of positive pairs, (N,)
            
            # Compute the dot products for negative pairs
            sim_neg = off_diagonal(dot_products)  # Similarity of negative pairs, (N, N-1)
            sim_neg_square = sim_neg.pow(2)  # Squared similarity of negative pairs, (N, N-1) 
            sim_neg_square = sim_neg_square.mean(dim=1)  # (N,)

        elif a.ndim == 3:
            # x is of shape (N, D), a is of shape (N, r, D)
            assert x.shape[0] == a.shape[0], "Input tensors must have the same batch size"
            assert x.shape[1] == a.shape[2], "Input tensors must have the same embedding dimension"
            r = a.shape[1]
            
            if self.exp_parameterization is None:
                # Z[i, j, k] = X[i]^T A[j, k]
                dot_products = torch.einsum('id,jkd->ijk', x, a)  # Shape: (N, N, r)
            elif self.exp_parameterization == "inner_product":
                dot_products = torch.einsum('id,jkd->ijk', x, a) / self.temparature  # Shape: (N, N, r)
                dot_products = torch.exp(dot_products)
            elif self.exp_parameterization == "squared":
                # Z[i, j, k] = exp(|| X[i] - A[j, k] ||^2 / T)
                squared_diff = ((x[:, None, None, :] - a[None, :, :, :]) ** 2).sum(dim=-1)  # Shape: (N, N, r)
                dot_products = torch.exp(squared_diff / self.temperature) # (N, N, r)

            # Compute the dot products for positive pairs
            sim_pos = digonal(dot_products)  # Similarity of positive pairs, (N,r)
            sim_pos = sim_pos.mean(dim=1)  # Average over r, resulting in (N,) 
            
            # Compute the dot products for negative pairs
            sim_neg = off_diagonal(dot_products)  # Similarity of negative pairs, (N, (N-1) * r)
            sim_neg_square = sim_neg.pow(2)  # Squared similarity of negative pairs, (N, (N-1) * r) 
            sim_neg_square = sim_neg_square.mean(dim=1)  # (N,)
            
        lora_loss = - 2 * sim_pos + sim_neg_square
        mean_sim_pos = sim_pos.mean()
        mean_sim_neg = sim_neg_square.mean()

        if reduction == "mean":
            lora_loss = lora_loss.mean()
            
        loss_dict = {"train/loss": torch.mean(lora_loss).item(), 
                     "train/mean_sim_pos": mean_sim_pos.item(), 
                     "train/mean_sim_neg": mean_sim_neg.item() }
        
        return lora_loss, loss_dict

def digonal(tensor: torch.Tensor) -> torch.Tensor:
    """
    If tensor is a 2D matrix of shape (n,n), return the diagonal elements in shape (n,)
    If tensor is a 3D matrix of shape (n, n, r), return the diagonal elements in shape (n, r).
    """
    n = tensor.shape[0]
    assert tensor.shape[0] == tensor.shape[1], "Input must be a square matrix"
    if tensor.ndim == 3:
        # Extract diagonal entries for 3D tensor
        idx = torch.arange(n)
        diag = tensor[idx, idx, :]
        return diag.view(n, -1)  # Reshape to (n, r)
    elif tensor.ndim == 2:
        # Extract diagonal entries for 2D tensor
        idx = torch.arange(n)
        diag = tensor[idx, idx]
        return diag.view(n)  # Reshape to (n,)

def off_diagonal(tensor: torch.Tensor) -> torch.Tensor:
    """
    If tensor is a 2D matrix of shape (n,n), return the off-diagonal elements of a square matrix in shape (n, n-1)
    If tensor is a 3D matrix of shape (n, n, r), return the off-diagonal elements of a square matrix in shape (n, (n-1) * r).
    """
    n = tensor.shape[0]
    assert tensor.shape[0] == tensor.shape[1], "Input must be a square matrix"
    
    # Create a mask that is True for off-diagonal elements
    mask = ~torch.eye(n, dtype=torch.bool, device=tensor.device)
    
    # Extract off-diagonal entries
    off_diag = tensor[mask]  # This flattens the first two dims into one
    off_diag = off_diag.view(n, -1) 
    
    return off_diag

if __name__ == "__main__":
    def compute_exponential_squared_distance(x: torch.Tensor, a: torch.Tensor, temperature: float) -> torch.Tensor:
        squared_diff = ((x[:, None, :] - a[None, :, :]) ** 2).sum(dim=-1)
        return torch.exp(squared_diff / temperature)

    def test_compute_exponential_squared_distance():
        x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # shape (2, 2)
        a = torch.tensor([[1.0, 0.0], [1.0, 1.0]])  # shape (2, 2)
        temperature = 1.0

        # Manually compute:
        # Z[0,0] = exp(||[1,0] - [1,0]||^2 / 1) = exp(0) = 1
        # Z[0,1] = exp(||[1,0] - [1,1]||^2 / 1) = exp(1)
        # Z[1,0] = exp(||[0,1] - [1,0]||^2 / 1) = exp(2)
        # Z[1,1] = exp(||[0,1] - [1,1]||^2 / 1) = exp(1)
        expected = torch.tensor([
            [torch.exp(torch.tensor(0.0)), torch.exp(torch.tensor(1.0))],
            [torch.exp(torch.tensor(2.0)), torch.exp(torch.tensor(1.0))]
        ])
        result = compute_exponential_squared_distance(x, a, temperature)
        print(result) 
        assert torch.allclose(result, expected), f"Expected {expected}, got {result}"

    test_compute_exponential_squared_distance()
