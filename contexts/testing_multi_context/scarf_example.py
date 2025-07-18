#!/usr/bin/env python3
"""
SCARF (Self-supervised Contrastive Learning using Random Feature Corruption) Example

This example demonstrates how to use the SCARF augmentation functionality
in the Contextures framework for self-supervised learning tasks.
"""

import argparse
import sys
import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path

# Add the parent directory to the path to import Contextures
sys.path.append(str(Path(__file__).parent.parent.parent))

from contexts.multi_conditional_context import (
    SCARFKnowledge, 
    MultiConditionalContext, 
    SCARFTransform
)
from utils.registry import get_transform


def create_sample_dataset(n_samples=1000, n_features=10, seed=42):
    """Create a sample dataset for demonstration."""
    np.random.seed(seed)
    
    # Create synthetic data with different feature distributions
    data = {}
    
    # Normal features
    data['normal_1'] = np.random.normal(0, 1, n_samples)
    data['normal_2'] = np.random.normal(5, 2, n_samples)
    
    # Uniform features
    data['uniform_1'] = np.random.uniform(-3, 3, n_samples)
    data['uniform_2'] = np.random.uniform(10, 20, n_samples)
    
    # Bimodal features
    data['bimodal_1'] = np.concatenate([
        np.random.normal(-2, 0.5, n_samples // 2),
        np.random.normal(2, 0.5, n_samples // 2)
    ])
    
    # Categorical-like features (continuous but clustered)
    data['categorical_like'] = np.random.choice([0, 5, 10], n_samples, p=[0.3, 0.4, 0.3])
    data['categorical_like'] += np.random.normal(0, 0.1, n_samples)
    
    # Add some noise features
    for i in range(n_features - 6):
        data[f'noise_{i}'] = np.random.normal(0, 1, n_samples)
    
    df = pd.DataFrame(data)
    
    # Add target variable (for demonstration)
    df['target'] = (df['normal_1'] + df['uniform_1'] + np.random.normal(0, 0.1, n_samples) > 0).astype(int)
    
    return df


def demonstrate_basic_usage():
    """Demonstrate basic SCARF usage."""
    print("=== Basic SCARF Usage Example ===")
    
    # Create sample dataset
    dataset = create_sample_dataset(n_samples=100, n_features=8)
    print(f"Dataset shape: {dataset.shape}")
    print(f"Features: {list(dataset.columns[:-1])}")  # Exclude target
    print()
    
    # 1. Using SCARFKnowledge directly
    print("1. Direct SCARFKnowledge usage:")
    config = {
        'distribution': 'uniform',
        'corruption_rate': 0.3,
        'uniform_eps': 1e-6
    }
    
    knowledge = SCARFKnowledge(config)
    knowledge.fit(dataset.drop('target', axis=1))
    
    # Convert to tensor
    x_tensor = torch.tensor(dataset.drop('target', axis=1).values, dtype=torch.float32)
    y_tensor = torch.tensor(dataset['target'].values, dtype=torch.float32)
    
    # Apply SCARF corruption
    x_corrupted, y = knowledge.transform(x_tensor, y_tensor)
    
    print(f"Original shape: {x_tensor.shape}")
    print(f"Corrupted shape: {x_corrupted.shape}")
    print(f"Corruption rate: {config['corruption_rate']}")
    print()
    
    # Show some statistics
    corruption_diff = torch.abs(x_tensor - x_corrupted)
    actual_corruption_rate = (corruption_diff > 1e-6).float().mean()
    print(f"Actual corruption rate: {actual_corruption_rate:.3f}")
    print()


def demonstrate_multi_conditional_context():
    """Demonstrate multi-conditional context for multiple augmentations."""
    print("=== Multi-Conditional Context Example ===")
    
    dataset = create_sample_dataset(n_samples=50, n_features=6)
    
    # Create knowledge component
    config = {
        'distribution': 'gaussian',
        'corruption_rate': 0.4,
    }
    knowledge = SCARFKnowledge(config)
    
    # Create multi-conditional context for 3 augmentations
    context = MultiConditionalContext(knowledge, num_augmentations=3)
    context.fit(dataset.drop('target', axis=1))
    
    # Convert to tensor
    x_tensor = torch.tensor(dataset.drop('target', axis=1).values, dtype=torch.float32)
    y_tensor = torch.tensor(dataset['target'].values, dtype=torch.float32)
    
    # Generate multiple augmentations
    x_augmented, y = context.transform(x_tensor, y_tensor)
    
    print(f"Original shape: {x_tensor.shape}")
    print(f"Augmented shape: {x_augmented.shape}")  # (batch_size, num_augmentations, num_features)
    print(f"Number of augmentations: {x_augmented.shape[1]}")
    print()
    
    # Show statistics for each augmentation
    for i in range(x_augmented.shape[1]):
        corruption_diff = torch.abs(x_tensor - x_augmented[:, i, :])
        corruption_rate = (corruption_diff > 1e-6).float().mean()
        print(f"Augmentation {i+1} corruption rate: {corruption_rate:.3f}")
    print()


def demonstrate_registry_usage():
    """Demonstrate using SCARF through the registry system."""
    print("=== Registry Usage Example ===")
    
    dataset = create_sample_dataset(n_samples=80, n_features=5)
    
    # Get SCARF transform from registry
    scarf_transform = get_transform('scarf')(
        distribution='bimodal',
        corruption_rate=0.5,
        num_augmentations=2
    )
    
    # Fit the transform
    scarf_transform.fit(dataset.drop('target', axis=1))
    
    # Convert to tensor
    x_tensor = torch.tensor(dataset.drop('target', axis=1).values, dtype=torch.float32)
    y_tensor = torch.tensor(dataset['target'].values, dtype=torch.float32)
    
    # Apply tensor transformation
    x_augmented, y = scarf_transform.transform_tensor(x_tensor, y_tensor)
    
    print(f"Original shape: {x_tensor.shape}")
    print(f"Augmented shape: {x_augmented.shape}")
    print(f"Distribution: bimodal")
    print(f"Corruption rate: 0.5")
    print()


def demonstrate_different_distributions():
    """Demonstrate different distribution types."""
    print("=== Distribution Comparison ===")
    
    dataset = create_sample_dataset(n_samples=60, n_features=4)
    x_tensor = torch.tensor(dataset.drop('target', axis=1).values, dtype=torch.float32)
    
    distributions = ['uniform', 'gaussian', 'bimodal']
    
    for dist in distributions:
        print(f"\n{dist.upper()} Distribution:")
        
        config = {
            'distribution': dist,
            'corruption_rate': 0.3,
        }
        
        knowledge = SCARFKnowledge(config)
        knowledge.fit(dataset.drop('target', axis=1))
        
        x_corrupted, _ = knowledge.transform(x_tensor)
        
        # Calculate statistics
        corruption_diff = torch.abs(x_tensor - x_corrupted)
        corruption_rate = (corruption_diff > 1e-6).float().mean()
        mean_diff = corruption_diff.mean()
        std_diff = corruption_diff.std()
        
        print(f"  Corruption rate: {corruption_rate:.3f}")
        print(f"  Mean difference: {mean_diff:.3f}")
        print(f"  Std difference: {std_diff:.3f}")


def demonstrate_self_supervised_learning():
    """Demonstrate SCARF in a self-supervised learning context."""
    print("=== Self-Supervised Learning Example ===")
    
    dataset = create_sample_dataset(n_samples=200, n_features=8)
    
    # Create SCARF context for contrastive learning
    config = {
        'distribution': 'uniform',
        'corruption_rate': 0.4,
    }
    knowledge = SCARFKnowledge(config)
    context = MultiConditionalContext(knowledge, num_augmentations=2)
    context.fit(dataset.drop('target', axis=1))
    
    # Simulate training loop
    x_tensor = torch.tensor(dataset.drop('target', axis=1).values, dtype=torch.float32)
    
    print("Simulating contrastive learning training loop:")
    for epoch in range(3):
        # Generate positive pairs (augmented versions)
        x_augmented, _ = context.transform(x_tensor)
        
        # x_augmented shape: (batch_size, 2, num_features)
        # Each sample has 2 augmented versions for contrastive learning
        
        # Simulate contrastive loss computation
        # In practice, you would pass these through an encoder and compute contrastive loss
        batch_size, num_augmentations, num_features = x_augmented.shape
        
        print(f"Epoch {epoch + 1}:")
        print(f"  Batch size: {batch_size}")
        print(f"  Augmentations per sample: {num_augmentations}")
        print(f"  Feature dimension: {num_features}")
        print(f"  Total augmented samples: {batch_size * num_augmentations}")
        print()


def main():
    """Main function to run all demonstrations."""
    parser = argparse.ArgumentParser(description='SCARF Augmentation Examples')
    parser.add_argument('--demo', choices=['basic', 'multi', 'registry', 'distributions', 'ssl', 'all'],
                       default='all', help='Which demonstration to run')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("SCARF (Self-supervised Contrastive Learning using Random Feature Corruption)")
    print("=" * 80)
    print()
    
    if args.demo == 'all' or args.demo == 'basic':
        demonstrate_basic_usage()
    
    if args.demo == 'all' or args.demo == 'multi':
        demonstrate_multi_conditional_context()
    
    if args.demo == 'all' or args.demo == 'registry':
        demonstrate_registry_usage()
    
    if args.demo == 'all' or args.demo == 'distributions':
        demonstrate_different_distributions()
    
    if args.demo == 'all' or args.demo == 'ssl':
        demonstrate_self_supervised_learning()
    
    print("=" * 80)
    print("SCARF demonstration completed!")


if __name__ == "__main__":
    main() 