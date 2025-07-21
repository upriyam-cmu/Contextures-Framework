import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

class SVDTrainer:
    def __init__(self, x_encoder, a_encoder, 
                optimizer, criterion, train_loader, 
                num_epochs, lr_scheduler = None,
                device='cuda'):
        """
        Initializes the Trainer.

        Args:
            x_encoder (nn.Module): NN encoder for inputs x
            a_encoder (nn.Module): NN encoder for contexts a 
            optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.
            criterion (nn.Module): The loss function defined in losses, e.g. SVD_LoRA
            train_loader (DataLoader): DataLoader for the training dataset.
            num_epochs (int): The total number of training epochs.
            device (str): The device to run training on ('cpu' or 'cuda').
        """
        self.x_encoder = x_encoder.to(device)
        self.a_encoder = a_encoder.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.num_epochs = num_epochs
        self.device = device
        self.lr_scheduler = None

    def _train_epoch(self, epoch):
        """
        Performs one epoch of training.
        """
        self.x_encoder.train()
        self.a_encoder.train()
        loss_recorder = {}

        for x, a in self.train_loader:
            x, a = x.to(self.device), a.to(self.device)
            # Forward pass
            x_embeddings = self.x_encoder(x)
            a_embeddings = self.a_encoder(a)
            loss, loss_dict = self.criterion(x_embeddings, a_embeddings, reduction = 'mean')
            
            # Backward and optimize
            self.optimizer.zero_grad() # Clear previous gradients
            loss.backward()            # Backpropagation
            self.optimizer.step()     # Update parameters

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            for k, v in loss_dict.items():
                if k not in loss_recorder:
                    loss_recorder[k] = 0.0
                loss_recorder[k] += v

        print(f"Epoch [{epoch+1}/{self.num_epochs}], ", end="")
        for k, v in loss_recorder.items():
            loss_recorder[k] /= len(self.train_loader)
            print(f"{k}: {loss_recorder[k]:.3f}, ", end="")
        print() 
        return loss_recorder

    def train(self):
        """
        Starts the training process.

        Args:
            model_save_path (str): Path to save the best performing model.
        """
        print("\nStarting training...")
        for epoch in range(self.num_epochs):
            loss_dicts = self._train_epoch(epoch)

        print("Training finished.")
    
    def save_model(self, model_save_dir):
        """
        Save the x_encoder and a_encoder
        """
        torch.save(self.x_encoder.state_dict(), f"{model_save_dir}/x_encoder.pth")
        torch.save(self.a_encoder.state_dict(), f"{model_save_dir}/a_encoder.pth")        

        