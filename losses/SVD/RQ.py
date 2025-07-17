import torch
from torch import nn
from typing import Sequence, Union, List, Literal
from torch.nn import functional as F

class SVDRQ(nn.Module):
    """
    RQ (Rayleigh-Quotient) loss implementation. 
    Naive version:
    L =  1/d E_{x,a ~ P(x,a)}[ || \Phi(x) - \Psi(a) ||_2^2 ] 
        + alpha / d \sum_{i} ( E_{x ~ p(x)}[ || \Phi_i(x) ||_2^2 ] - 1 )^2
        + beta / d(d-1)  \sum_{i \neq j} E_{x ~p(x)} [ \Phi_i(x) \Phi_j(x) ]^2,
    where \Phi(x) and \Psi(a) are the embeddings of inputs x and contexts a, respectively.
    """
    def __init__(self,
                 x_proj: nn.Module = None, 
                 a_proj: nn.Module = None,
                 alpha: float = 10.0,
                 beta: float = 30.0, 
                 ):
        """
        Initialize the RQ loss module.
        Args:
        - x_proj: a MLP module that further projects inputs x to embeddings. \Phi'(x) = x_proj(\Phi(x))
        - a_proj: a MLP module that further projects contexts a to embeddings. \Psi'(a) = a_proj(\Psi(a))
        - alpha (float): Weight for variance term.
        - beta (float): Weight for covariance term.
        """
        super(SVDRQ, self).__init__()

        self.x_proj = x_proj
        self.a_proj = a_proj
        self.alpha = alpha  # variance coefficient
        self.beta = beta    # covariance coefficient

    def forward(self, x: torch.Tensor, a: torch.Tensor,
                reduction: Literal["mean", "none"] = "mean",
                ) -> torch.Tensor:
        """
        Inputs:
        - x: embedding of inputs x, torch tensor of shape (N, D)
        - a: embedding of contexts a, torch tensor of shape (N,D) or (N, r, D), representing single context or r contexts for each input 
        
        Outputs:
        - RQ_loss: RQ loss, torch tensor of shape (N,) or scalar if mean reduction is applied.
        - loss_dict: Dictionary containing verbose information, containing the main loss, variance loss, and covariance loss.
        """
        N, D = x.shape  # Batch size and embedding dimension
        if x_proj is not None:
            x = self.x_proj(x)
        if a_proj is not None:
            if a.ndim == 2:
                a = self.a_proj(a)
            elif a.ndim == 3:
                N, r, D = a.shape
                a = self.a_proj(a.view(N*r, D)).view(N, r, -1)

        # Main term: Mean squared error between embeddings
        invariance_loss = torch.mean( ((x-a) ** 2), dim=1 ) # divided by dimension, (N,)
       
        # Regularization terms are only implemented for x
        # diagonal term: C_ii = (E_x[f_i(x)^2]  -1)^2 = E_x[f_i(x)^2] * E_x[f_i(x)^2] - 2 * E_x[f_i(x)^2] + 1
        # split a batch into two halves
        x1, x2 = x.chunk(2, dim=0)
        
        # Unbiased estimator of inner product
        C_ii = torch.mean(x1**2, dim=0) * torch.mean(x2**2, dim=0) - 2 * torch.mean(x**2, dim=0) + 1 # (D,)
        diagonal_loss = torch.mean(C_ii) 

        # off-diagonal term: Reduce off-diagonal values in covariance matrix
        # C_ij = E_x[f_i(x)f_j(x)]^2 = E_x[f_i(x)f_j(x)] * E_x[f_i(x)f_j(x)] 
        cov_z11 = (x1.T @ x1) / x1.shape[0]  # (D,D)
        cov_z12 = (x2.T @ x2) / x2.shape[0]  # (D,D)
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
