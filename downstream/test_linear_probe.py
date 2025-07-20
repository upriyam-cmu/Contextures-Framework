#!/usr/bin/env python3
"""
Test script for linear probe implementation
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent))

from downstream.linear_probe import LinearProbe, train_linear_probe, extract_features
from encoders.mlp import MLPEncoder

def test_linear_probe_classification():
    """Test linear probe for classification task"""
    print("Testing Linear Probe - Classification...")
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 64
    
    # Create features (simulating extracted features from encoder)
    features = np.random.normal(0, 1, (n_samples, n_features))
    
    # Create synthetic targets (3 classes)
    targets = np.random.choice([0, 1, 2], n_samples, p=[0.4, 0.3, 0.3])
    
    # Test the train_linear_probe function
    probe, results = train_linear_probe(
        features=features,
        targets=targets,
        task_type="classification",
        test_size=0.2,
        val_size=0.2,
        max_epochs=50,
        batch_size=128,
        verbose=False
    )
    
    print(f"Classification Results:")
    print(f"  Accuracy: {results['test_metrics']['accuracy']:.4f}")
    print(f"  Train size: {results['data_splits']['train_size']}")
    print(f"  Val size: {results['data_splits']['val_size']}")
    print(f"  Test size: {results['data_splits']['test_size']}")
    print(f"  Epochs trained: {results['train_history']['epochs_trained']}")
    
    # Test direct probe usage
    probe_direct = LinearProbe(
        input_dim=n_features,
        task_type="classification",
        num_classes=3,
        max_epochs=20
    )
    
    # Split data manually
    train_size = int(0.8 * n_samples)
    X_train, X_test = features[:train_size], features[train_size:]
    y_train, y_test = targets[:train_size], targets[train_size:]
    
    # Train
    history = probe_direct.fit(X_train, y_train, verbose=False)
    
    # Evaluate
    metrics = probe_direct.evaluate(X_test, y_test)
    print(f"  Direct probe accuracy: {metrics['accuracy']:.4f}")
    
    return True

def test_linear_probe_regression():
    """Test linear probe for regression task"""
    print("\nTesting Linear Probe - Regression...")
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 32
    
    # Create features
    features = np.random.normal(0, 1, (n_samples, n_features))
    
    # Create synthetic targets (continuous)
    targets = np.sum(features * np.random.normal(0, 1, n_features), axis=1) + np.random.normal(0, 0.1, n_samples)
    
    # Test the train_linear_probe function
    probe, results = train_linear_probe(
        features=features,
        targets=targets,
        task_type="regression",
        test_size=0.2,
        val_size=0.2,
        max_epochs=50,
        batch_size=128,
        verbose=False
    )
    
    print(f"Regression Results:")
    print(f"  R² Score: {results['test_metrics']['r2_score']:.4f}")
    print(f"  MSE: {results['test_metrics']['mse']:.4f}")
    print(f"  RMSE: {results['test_metrics']['rmse']:.4f}")
    
    return True

def test_feature_extraction():
    """Test feature extraction from encoder"""
    print("\nTesting Feature Extraction...")
    
    # Create a mock encoder
    encoder = MLPEncoder(input_dim=10, output_dim=64, hidden_dims=[32, 32])
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    
    # Create data
    data = np.random.normal(0, 1, (n_samples, n_features))
    targets = np.random.choice([0, 1], n_samples)
    
    # Create DataLoader
    from torch.utils.data import TensorDataset, DataLoader
    dataset = TensorDataset(torch.tensor(data, dtype=torch.float32), 
                           torch.tensor(targets, dtype=torch.long))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Extract features
    features, extracted_targets = extract_features(encoder, dataloader, device="cpu")
    
    print(f"Feature Extraction Results:")
    print(f"  Features shape: {features.shape}")
    print(f"  Targets shape: {extracted_targets.shape}")
    print(f"  Expected output dim: 64")
    print(f"  Actual output dim: {features.shape[1]}")
    
    assert features.shape[1] == 64, f"Expected 64 features, got {features.shape[1]}"
    assert features.shape[0] == n_samples, f"Expected {n_samples} samples, got {features.shape[0]}"
    
    return True

def test_tensor_dataframe_compatibility():
    """Test compatibility with different data types"""
    print("\nTesting Data Type Compatibility...")
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 100
    n_features = 16
    
    features_np = np.random.normal(0, 1, (n_samples, n_features))
    targets_np = np.random.choice([0, 1], n_samples)
    
    # Test with numpy arrays
    probe_np = LinearProbe(input_dim=n_features, task_type="classification", num_classes=2, max_epochs=5)
    history_np = probe_np.fit(features_np, targets_np, verbose=False)
    
    # Test with pandas DataFrames
    features_df = pd.DataFrame(features_np, columns=[f"feature_{i}" for i in range(n_features)])
    targets_df = pd.Series(targets_np)
    
    probe_df = LinearProbe(input_dim=n_features, task_type="classification", num_classes=2, max_epochs=5)
    history_df = probe_df.fit(features_df, targets_df, verbose=False)
    
    # Test with torch tensors
    features_tensor = torch.tensor(features_np, dtype=torch.float32)
    targets_tensor = torch.tensor(targets_np, dtype=torch.long)
    
    probe_tensor = LinearProbe(input_dim=n_features, task_type="classification", num_classes=2, max_epochs=5)
    history_tensor = probe_tensor.fit(features_tensor, targets_tensor, verbose=False)
    
    print(f"Data Type Compatibility Results:")
    print(f"  NumPy arrays: ✓")
    print(f"  Pandas DataFrames: ✓")
    print(f"  PyTorch tensors: ✓")
    
    return True

def main():
    """Run all tests"""
    print("Running Linear Probe Tests...\n")
    
    try:
        test_linear_probe_classification()
        test_linear_probe_regression()
        test_feature_extraction()
        test_tensor_dataframe_compatibility()
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main() 