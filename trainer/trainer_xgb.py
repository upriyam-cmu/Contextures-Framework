import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from encoders.xgboost import XGBEncoderBase

from typing import Optional
from sklearn.decomposition import PCA

class SVDTrainerXGB:
    def __init__(self, x_encoder: XGBEncoderBase, a_encoder: XGBEncoderBase,
                encoding_dim: int, criterion: nn.Module, train_loader: DataLoader,
                num_epochs: int, record_losses: bool = True, lr_scheduler = None,
                device: str = 'cuda'):
        """
        Initializes the Trainer.

        Args:
            x_encoder (XGBEncoderBase): NN encoder for inputs x
            a_encoder (XGBEncoderBase): NN encoder for contexts a 
            optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.
            criterion (nn.Module): The loss function defined in losses, e.g. SVD_LoRA
            train_loader (DataLoader): DataLoader for the training dataset.
            num_epochs (int): The total number of training epochs.
            device (str): The device to run training on ('cpu' or 'cuda').
        """
        self.x_encoder = x_encoder.train(criterion)
        self.a_encoder = a_encoder.train(criterion)
        self.criterion = criterion
        self.train_loader = train_loader
        self.num_epochs = num_epochs
        self.device = device
        self.lr_scheduler = None

        self.record_losses = record_losses
        self.encoding_dim = encoding_dim
        self.warm_start_encoder = PCA(n_components=encoding_dim)

    def _warm_start_target(self, x: np.ndarray) -> np.ndarray:
        batch_size, n_fts = x.shape
        n_rep_bsz = (2 * self.encoding_dim - 1) // batch_size + 1
        n_rep_fts = (2 * self.encoding_dim - 1) // n_fts + 1
        ret = self.warm_start_encoder.fit_transform(np.tile(x, (n_rep_bsz, n_rep_fts)))[:batch_size]
        assert ret.shape == (batch_size, self.encoding_dim)
        return ret

    def _train_warm_start(self):
        for x, a in self.train_loader:
            x, a = x.detach().cpu().numpy(), a.detach().cpu().numpy()

            a = a.reshape(a.shape[0], -1)  # flatten contexts?

            self.a_encoder.fit_batch(a, self._warm_start_target(x))
            self.x_encoder.fit_batch(x, self._warm_start_target(a))

    def _x_out(self, x: np.ndarray) -> np.ndarray:
        return self.x_encoder.transform(x, pred_leaf=False)

    def _a_out(self, a: np.ndarray) -> np.ndarray:
        return self.a_encoder.transform(a, pred_leaf=False)

    def _train_epoch(self, epoch: int):
        """
        Performs one epoch of training.
        """
        loss_recorder = {}

        for x, a in self.train_loader:
            x, a = x.detach().cpu().numpy(), a.detach().cpu().numpy()

            a = a.reshape(a.shape[0], -1)  # flatten contexts?

            self.a_encoder.fit_batch(a, self._x_out(x))
            self.x_encoder.fit_batch(x, self._a_out(a))

            # bookkeeping
            if self.record_losses:
                _, loss_dict = self.criterion(
                    torch.tensor(self._x_out(x), device=self.device),
                    torch.tensor(self._a_out(a), device=self.device),
                    reduction='mean',
                )

                for k, v in loss_dict.items():
                    if k not in loss_recorder:
                        loss_recorder[k] = 0.0
                    loss_recorder[k] += v

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        print(f"Epoch [{epoch+1}/{self.num_epochs}], ", end="")
        for k, v in loss_recorder.items():
            loss_recorder[k] /= len(self.train_loader)
            print(f"{k}: {loss_recorder[k]:.3f}, ", end="")
        print() 
        return loss_recorder

    def train(self):
        """
        Starts the training process.

        """
        print("\nStarting training...")
        print("Step 1/2: XGB Warm Start")
        self._train_warm_start()

        print("Step 2/2: Training Loop")
        for epoch in range(self.num_epochs):
            loss_dicts = self._train_epoch(epoch)

        print("Training finished.")
    
    def save_model(self, model_save_dir):
        """
        Save the x_encoder and a_encoder
        """
        raise NotImplementedError()
