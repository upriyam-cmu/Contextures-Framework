# -*- coding: utf-8 -*-
import os, sys
import numpy as np
import torch
import json
import yaml
from tqdm import tqdm
from easydict import EasyDict as edict
from torch.utils.data import DataLoader
from utils.registry import get_encoder, get_loss, get_context

from trainer.trainer import SVDTrainer
from downstream.linear_probe import train_linear_probe, extract_features

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return edict(config)

def main(config):
    # Global setting
    torch.manual_seed(config.global.seed)
    np.random.seed(config.global.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create directories
    results_dir = config.global.results_dir
    model_save_dir = os.path.join(results_dir, "models")
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Load dataset 
    """
    TODO for Arnav: Load cls42/cls56 datasets using existing splits (train/val/test) 
    The data format should be in pandas so that we can apply feature transformation later
    """

    # Feature transformation
    """
    TODO for Arnav: Apply feature transformation pipeline to inputs x
    At this stage, we want to have a pytorch and dataframe dataset, e.g. TabularDataset in https://github.com/RuntianZ/TabularRL/blob/main/framework/dataset.py
    """
    train_df, train_dataset = None, None
    val_df, val_dataset = None, None
    test_df, test_dataset = None, None

    # Get contexts
    """
    TODO for Hugo: finish transform_multiple function in scarf.py
    """
    context_class = get_context(config.context.name)
    context = context_class(**config.context.parameters)
    context.fit(train_df)
    
    # Create dataloader
    collate_fn = context.get_collate_fn() 
    train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True, collate_fn=collate_fn)

    # Get encoders: TODO: Get input dim from train_dataset
    encoder_class = get_encoder(config.encoder.name)
    x_encoder = encoder_class(input_dim = None, **config.encoder.parameters).to(device)
    a_encoder = encoder_class(input_dim = context.context_dim, **config.encoder.parameters).to(device)

    # Set optimizers: use adam without scheduler for now
    optimizer = torch.optim.Adam([x_encoder.parameters(), a_encoder.parameters()], lr=config.train.lr)

    # Loss function
    loss_class = get_loss(config.losses.name)
    criterion = loss_class(**config.losses.parameters)
    
    # Trainer: use SVDTrainer for testing
    trainer = SVDTrainer(x_encoder, a_encoder, 
                optimizer, criterion, train_loader, 
                config.train.num_epochs,
                device=device)
    
    trainer.train()
    trainer.save_model(model_save_dir)

    # Test downstream performance
    """
    TODO for Hugo: Add linear probe function at here and downstream.linear_probe.py
    1. Iteratre train dataset and use the x_encoder to extract train features
    2. Train a linear probe on top of it
    3. Test the performance on val/test set
    """
    print("\n=== Linear Probe Evaluation ===")
    
    # 1. Extract features from train dataset using x_encoder
    print("Extracting features from training data...")
    train_features, train_targets = extract_features(x_encoder, train_loader, device=device)
    
    # 2. Train linear probe on extracted features
    print("Training linear probe...")
    probe, results = train_linear_probe(
        features=train_features,
        targets=train_targets,
        task_type="classification",
        test_size=0.2,
        val_size=0.2,
        max_epochs=50,
        batch_size=128,
        verbose=False
    )
    
    # 3. Test performance on val/test set
    print("Linear Probe Results:")
    print(f"  Test Accuracy: {results['test_metrics']['accuracy']:.4f}")
    print(f"  Train size: {results['data_splits']['train_size']}")
    print(f"  Val size: {results['data_splits']['val_size']}")
    print(f"  Test size: {results['data_splits']['test_size']}")
    print(f"  Epochs trained: {results['train_history']['epochs_trained']}")
    
    # Save probe results
    probe_results_path = os.path.join(results_dir, "linear_probe_results.json")
    with open(probe_results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Linear probe results saved to: {probe_results_path}")


if __name__ == '__main__':
    #config_path = sys.argv[1]
    config_path = "config.yaml"
    config = load_config(config_path)
    main(config)
