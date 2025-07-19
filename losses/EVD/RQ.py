import torch
from torch import nn
from typing import Sequence, Union, List, Literal
from torch.nn import functional as F
from utils.registry import register_loss

@register_loss('EVDRQ')
class EVDRQ(nn.Module):
    """
    RQ (Rayleigh-Quotient) loss implementation. 
    Naive version:
    L =  1/d E_{a, a' ~ P(a,a')}[ || \Psi(a) - \Psi(a') ||_2^2 ] 
        + alpha / d \sum_{i} ( E_{a ~ p(a)}[ || \Psi_i(a) ||_2^2 ] - 1 )^2
        + beta / d(d-1)  \sum_{i \neq j} E_{a ~p(a)} [ \Psi_i(a) \Psi_j(a) ]^2,
    where \Psi(a) are the embeddings of contexts a, and d is the dimension of embedding.
    """
    def __init__(self,
                 a_proj: nn.Module = None,
                 alpha: float = 10.0,
                 beta: float = 30.0, 
                 ):
        """
        Initialize the RQ loss module.
        Args:
        - a_proj: a MLP module that further projects contexts a to embeddings. \Psi'(a) = a_proj(\Psi(a))
        - alpha (float): Weight for variance term.
        - beta (float): Weight for covariance term.
        """
        super(EVDRQ, self).__init__()

        self.a_proj = a_proj
        self.alpha = alpha  # variance coefficient
        self.beta = beta    # covariance coefficient

    def forward(self, a: torch.Tensor,
                reduction: Literal["mean", "none"] = "mean",
                ) -> torch.Tensor:
        """
        Inputs:
        - a: embedding of contexts a, torch tensor of shape (N, r, D), representing single context or r contexts for each input (r >=2)
        
        Outputs:
        - RQ_loss: RQ loss, torch tensor of shape (N,) or scalar if mean reduction is applied.
        - loss_dict: Dictionary containing verbose information, containing the main loss, variance loss, and covariance loss.
        """
        if a_proj is not None:
            N, r, D = a.shape  # Batch size and embedding dimension
            a = self.a_proj(a.view(N*r, D)).view(N, r, -1)
        N, r, D = a.shape  # Batch size and embedding dimension after projection
        assert r >= 2, "Should have at least one pair."

        # Main term: Mean squared error between embeddings
        # Z[i, j, k] = || a[i,j] -  a[i,k] ||_2^2, for all j != k
        squared_diff = ((a[:, None, :, :] - a[:, :, None, :]) ** 2).sum(dim=-1)  # Shape: (N, r, r)
        mask_ii = torch.eye(r, dtype=torch.bool) # Shape: (r, r)
        squared_diff = squared_diff[:, mask_ii].view(N, -1) # Shape (N, r(r-1))
        invariance_loss = squared_diff.mean(dim=1) 

        # Regularization terms are only implemented for a
        # diagonal term: C_ii = (E_x[f_i(a)^2]  -1)^2 = E_a[f_i(a)^2] * E_a[f_i(a)^2] - 2 * E_a[f_i(a)^2] + 1
        # split a batch into two halves
        a1, a2 = a.chunk(2, dim=0)
        a1 = a1.view(-1, D)
        a2 = a2.view(-1, D)
        a_reshaped = a.view(-1,D)
        
        # Unbiased estimator of inner product
        C_ii = torch.mean(a1**2, dim=0) * torch.mean(a2**2, dim=0) - 2 * torch.mean(a_reshaped ** 2, dim=0) + 1 # (D,)
        diagonal_loss = torch.mean(C_ii) 

        # off-diagonal term: Reduce off-diagonal values in covariance matrix
        # C_ij = E_a[f_i(a)f_j(a)]^2 = E_a[f_i(a)f_j(a)] * E_a[f_i(a)f_j(a)] 
        cov_z11 = (a1.T @ a1) / a1.shape[0]  # (D,D)
        cov_z12 = (a2.T @ a2) / a2.shape[0]  # (D,D)
        C_ij = cov_z11 * cov_z12 # (D,D)
        off_diagonal_loss = off_diagonal(C_ij).mean() 
        
        RQ_loss = invariance_loss + self.alpha * diagonal_loss + self.beta * off_diagonal_loss
        
        loss_dict = {
            'train/total_loss': loss.mean().item(),
            'train/invariance_loss': invariance_loss.mean().item(),
            'train/diagonal_loss': diagonal_loss.item(),
            'train/off_diagonal_loss': off_diagonal_loss.item()
        }

        if reduction == "mean":
            RQ_loss = RQ_loss.mean()
        
        return RQ_loss, loss_dict
