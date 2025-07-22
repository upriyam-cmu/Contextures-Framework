import torch
from torch import nn
from typing import Sequence, Union, List, Literal
import numpy as np
from torch.nn import functional as F

from utils.registry import register_loss

@register_loss('EVDLoRA')
class EVDLoRA(nn.Module):
    r"""
    LoRA (Low-Rank Approximation) loss implementation.
    Naive version:
    L = -2 E_{a,a' ~ P(a,a')} [\Psi(a)^T \Psi(a')] + E_{a,a' ~ p(a)} [ (\Psi(a)^T \Psi(a') )^2].
    where \Psi(a) are the embeddings of contexts a
    
    Exponential parameterization version (inner product):
    L = -2 E_{a,a' ~ P(a,a')} [exp(\Psi(a)^T \Psi(a') / T )] + E_{a,a' ~ p(a)} [ ( exp(\Psi(a)^T \Psi(a') / T)  )^2].
    
    Exponential parameterization version (squared):
    L = -2 E_{a,a' ~ P(a,a')} [exp(|| \Psi(a) - \Psi(a') ||^2 / T )] + E_{a,a' ~ p(a)} [ || exp(\Psi(a) - \Psi(a') ||^2) / T) )^2].

    """
    def __init__(self, 
                 a_proj: nn.Module = None,
                 exp_parameterization: Literal["inner_product", "squared"] = None,
                 temperature: float = 1.0,
                 ) -> None:
        r"""
        Initialize the LoRA loss module.
        Args:
        - a_proj: a MLP module that further projects contexts a to embeddings. \Psi'(a) = a_proj(\Psi(a))
        - exp_paramerization:  whether to use exponential parameterization. 
        - temperature: float, temperature for exp_parameterization, default is 1.0.
        """
        super(EVDLoRA, self).__init__()
        self.a_proj = a_proj
        self.exp_parameterization = exp_parameterization
        self.temperature = temperature

    def forward(self, *args: torch.Tensor,
                reduction: Literal["mean", "none"] = "mean",
                ) -> torch.Tensor:
        r"""
        Inputs:
        - a: embedding of contexts a, torch tensor of shape (N, r, D), representing single context or r contexts for each input (r >=2)
        
        Outputs:
        - lora_loss: LoRA loss, torch tensor of shape (N,) or scalar if mean reduction is applied.
        - loss_dict: Dictionary containing verbose information like the loss of the positive pairs and negative pairs.
        """
        # accept either (a,) for (x, a)
        if len(args) == 1:
            a = args[0]
        elif len(args) == 2:
            _, a = args
        else:
            raise ValueError("LoRA loss expects 1 or 2 tensors, (got %d)" % len(args))
        
        if self.a_proj is not None:
            N, r, D = a.shape  # Batch size and embedding dimension
            a = self.a_proj(a.view(N*r, D)).view(N, r, -1)
        N, r, D = a.shape  # Batch size and embedding dimension after projection
        assert r>=2, "Should have at least one pair."

        if self.exp_parameterization is None:
            # Z[i, j, k, l] = A[i,j]^T A[k,l]
            dot_products = torch.einsum('ijd,kld->ikjl', a, a) # (N, N, r, r)
        elif self.exp_parameterization == "inner_product":
            sim = torch.einsum('ijd,kld->ikjl', a, a) / self.temperature # Shape: (N, N, r, r)
            sim_max = sim.detach().max(dim = -1, keepdim = True).values
            dot_products = torch.exp(sim - sim_max) # Shape: (N, N, r, r)
        elif self.exp_parameterization == "squared":
            # Z[i, j, k, l] = exp(|| A[i,j] - A[k, l] ||^2 / T)
            squared_diff = ((a[:, :, None, None, :] - a[None, None, :, :, :]) ** 2).sum(dim = -1)  # Shape: (N, N, r, r)
            dot_products = torch.exp(squared_diff / self.temperature) # (N, N, r, r)

        # Compute the dot products for positive pairs, dot_product[i,i,j,k] with j != k.
        mask_ii = torch.eye(N, dtype=torch.bool) # Shape: (N, N)
        # We need to create a mask that is True where j != k.
        mask_jk = ~torch.eye(r, dtype=torch.bool) # Shape: (r, r)

        # We need to broadcast mask_ii to (N, N, 1, 1) and mask_jk to (1, 1, r, r).
        final_mask = mask_ii.unsqueeze(2).unsqueeze(3) & mask_jk.unsqueeze(0).unsqueeze(1)

        sim_pos = dot_products[final_mask].view(N,-1)  # Similarity of positive pairs, (N,r*(r-1))
        sim_pos = sim_pos.mean(dim=1)  # Average over r * r, resulting in (N,) 
        
        # Compute the dot products for negative pairs, dot_product[i,j,k,l] with i !=j
        mask_neg = ~torch.eye(N, dtype=torch.bool) # Shape: (N, N)
        sim_neg = dot_products[mask_neg, :, :].view(N, -1) # Shape (N, N-1, r, r) -> (N, (N-1)*r*r)

        sim_neg_square = torch.square(sim_neg).clamp(min = 1e-8)  # Squared similarity of negative pairs (N, (N-1)*r*r) 
        sim_neg_square = sim_neg_square.view(N, -1).mean(dim=1)  # (N, (N-1)*r*r)
        
        lora_loss = -2.0 * sim_pos + sim_neg_square
        mean_sim_pos = sim_pos.mean()
        mean_sim_neg = sim_neg_square.mean()

        if reduction == "mean":
            lora_loss = lora_loss.mean()
            
        loss_dict = {"train/loss": torch.mean(lora_loss).item(), 
                     "train/mean_sim_pos": mean_sim_pos.item(), 
                     "train/mean_sim_neg": mean_sim_neg.item() }
        
        return lora_loss, loss_dict


if __name__ == "__main__":
    # Define dimensions for the tensor
    N = 3  # Example size for the first two dimensions
    r = 4  # Example size for the last two dimensions

    # Create a sample PyTorch tensor
    # We'll use torch.randn for random values, but you can replace this
    # with your actual tensor data.
    T = torch.randn(N, N, r, r)

    print(f"Original tensor shape: {T.shape}")
    mask_ii = torch.eye(N, dtype=torch.bool) # Shape: (N, N)
    Z = T[~mask_ii, :, :]
    print(Z.shape)
    exit()

    # --- Step 1: Create a mask for the first two dimensions (i, i) ---
    # We want elements where the first index equals the second index.
    # torch.eye(N, dtype=torch.bool) creates an identity matrix of shape (N, N)
    # with True on the diagonal and False elsewhere.

    print(f"Mask for (i, i) shape: {mask_ii.shape}")
    # Example:
    # If N=3, mask_ii would be:
    # [[ True, False, False],
    #  [False,  True, False],
    #  [False, False,  True]]

    # --- Step 2: Create a mask for the last two dimensions (j, k) where j != k ---
    # torch.eye(r, dtype=torch.bool) creates an identity matrix of shape (r, r).
    # We then invert it using the ~ operator to get True where j != k.
    mask_jk = ~torch.eye(r, dtype=torch.bool) # Shape: (r, r)
    print(f"Mask for (j, k) where j != k shape: {mask_jk.shape}")
    # Example:
    # If r=3, torch.eye(3) is:
    # [[ True, False, False],
    #  [False,  True, False],
    #  [False, False,  True]]
    # ~torch.eye(3) is:
    # [[False,  True,  True],
    #  [ True, False,  True],
    #  [ True,  True, False]]

    # --- Step 3: Combine the masks using broadcasting ---
    # To combine them, we need to add singleton dimensions to allow PyTorch's
    # broadcasting mechanism to work correctly.
    # mask_ii needs to be (N, N, 1, 1) to broadcast across r and r.
    # mask_jk needs to be (1, 1, r, r) to broadcast across N and N.

    # The unsqueeze(dim) method adds a new dimension of size 1 at the specified position.
    # mask_ii.unsqueeze(2).unsqueeze(3) -> (N, N, 1, 1)
    # mask_jk.unsqueeze(0).unsqueeze(1) -> (1, 1, r, r)
    final_mask = mask_ii.unsqueeze(2).unsqueeze(3) & mask_jk.unsqueeze(0).unsqueeze(1)

    # The '&' operator performs element-wise logical AND.
    # The resulting `final_mask` will have the shape (N, N, r, r).
    print(f"Final combined mask shape: {final_mask.shape}")

    # --- Step 4: Apply the mask to the tensor ---
    # When a boolean tensor is used for indexing, PyTorch selects all elements
    # from the original tensor where the corresponding boolean mask value is True.
    # The result will be a 1D tensor containing the extracted elements.
    extracted_elements = T[final_mask]
    print("extracted shape:",extracted_elements.shape)

    print(f"Shape of the extracted elements tensor: {extracted_elements.shape}")
    print("\nFirst 10 extracted elements (if available):")
    print(extracted_elements[:10])

    # --- Verification (Optional): Using a loop to compare ---
    # This loop is for verification purposes and demonstrates the logic.
    # For large tensors, the masked indexing above is significantly more efficient.
    extracted_elements_loop = []
    for i in range(N):
        for j in range(r):
            for k in range(r):
                if j != k:
                    # We are implicitly checking for i==i by fixing the first two indices to 'i'
                    extracted_elements_loop.append(T[i, i, j, k])

    # Convert the list of tensors to a single tensor
    extracted_elements_loop_tensor = torch.stack(extracted_elements_loop)

    print(f"\nShape of elements extracted via loop: {extracted_elements_loop_tensor.shape}")

    # Check if the results from both methods are the same (order might differ, so we sort)
    # For floating point numbers, it's best to use torch.allclose for comparison.
    # However, since we're just extracting, and not performing computations,
    # sorting and then checking for exact equality is fine here.
    print(f"Are the extracted elements from both methods equal (after sorting)? "
        f"{torch.equal(torch.sort(extracted_elements).values, torch.sort(extracted_elements_loop_tensor).values)}")
