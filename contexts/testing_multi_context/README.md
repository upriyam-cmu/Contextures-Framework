# SCARF Augmentation in Contextures

## Overview

SCARF (Self-supervised Contrastive Learning using Random Feature Corruption) is a powerful data augmentation technique for tabular data that works by randomly corrupting features with values sampled from their marginal distributions. This implementation is integrated into the Contextures framework and provides a flexible, extensible solution for self-supervised learning tasks.

## Quick Test

To quickly test if SCARF is working, run:

```bash
cd Contextures/contexts/testing_multi_context && python quick_test.py
```

## Features

- **Multiple Distribution Types**: Support for uniform, gaussian, and bimodal distributions
- **Configurable Corruption Rate**: Control the percentage of features to corrupt (0.0 to 1.0)
- **Multi-Augmentation Support**: Generate multiple augmented versions for contrastive learning
- **Flexible Output Formats**: Tensor, DataFrame, and NumPy array outputs
- **Registry Integration**: Seamless integration with the Contextures transform registry
- **Command-Line Interface**: Easy-to-use CLI for batch processing
- **Reproducible Results**: Configurable random seeds for consistent results

## Installation

The SCARF functionality is part of the Contextures package. Ensure you have the required dependencies:

```bash
pip install torch pandas numpy
```

## Quick Start

### Basic Python Usage

```python
import torch
import pandas as pd
from contexts.multi_conditional_context import SCARFKnowledge

# Create sample data
data = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5],
    'feature2': [10, 20, 30, 40, 50],
    'target': [0, 1, 0, 1, 0]
})

# Configure SCARF
config = {
    'distribution': 'uniform',
    'corruption_rate': 0.3,
    'uniform_eps': 1e-6
}

# Create and fit SCARF knowledge
knowledge = SCARFKnowledge(config)
knowledge.fit(data.drop('target', axis=1))

# Convert to tensor and apply augmentation
x_tensor = torch.tensor(data.drop('target', axis=1).values, dtype=torch.float32)
x_corrupted, y = knowledge.transform(x_tensor)

print(f"Original shape: {x_tensor.shape}")
print(f"Corrupted shape: {x_corrupted.shape}")
```

### Multi-Augmentation for Contrastive Learning

```python
from contexts.multi_conditional_context import MultiConditionalContext

# Create multi-conditional context for 3 augmentations
context = MultiConditionalContext(knowledge, num_augmentations=3)
context.fit(data.drop('target', axis=1))

# Generate multiple augmented versions
x_augmented, y = context.transform(x_tensor)
print(f"Augmented shape: {x_augmented.shape}")  # (batch_size, 3, num_features)
```

### Using the Registry System

```python
from utils.registry import get_transform

# Get SCARF transform from registry
scarf_transform = get_transform('scarf')(
    distribution='gaussian',
    corruption_rate=0.4,
    num_augmentations=2
)

# Fit and transform
scarf_transform.fit(data.drop('target', axis=1))
x_augmented, y = scarf_transform.transform_tensor(x_tensor)
```

## Command Line Interface

### Basic Commands

```bash
# Basic augmentation with default settings
python cli_scarf.py --input sample_data.csv --output augmented_data.csv

# Multiple augmentations with custom settings
python cli_scarf.py --input sample_data.csv --output augmented_data.csv \
  --distribution gaussian --corruption-rate 0.4 --num-augmentations 3

# Save as tensor format for deep learning
python cli_scarf.py --input sample_data.csv --output augmented_data.pt \
  --output-format tensor --num-augmentations 2

# Exclude target column from augmentation
python cli_scarf.py --input sample_data.csv --output augmented_data.csv \
  --target-column target --corruption-rate 0.3

# Use bimodal distribution
python cli_scarf.py --input sample_data.csv --output augmented_data.csv \
  --distribution bimodal --corruption-rate 0.5
```

### Advanced CLI Options

```bash
# Verbose output with detailed information
python cli_scarf.py --input sample_data.csv --output augmented_data.csv --verbose

# Save configuration for reuse
python cli_scarf.py --input sample_data.csv --output augmented_data.csv \
  --save-config scarf_config.json

# Load configuration from file
python cli_scarf.py --input sample_data.csv --output augmented_data.csv \
  --load-config scarf_config.json

# Set random seed for reproducibility
python cli_scarf.py --input sample_data.csv --output augmented_data.csv --seed 123
```

## API Reference

### SCARFKnowledge

The core SCARF implementation that handles feature corruption.

#### Parameters

- `config` (dict): Configuration dictionary containing:
  - `distribution` (str): Distribution type ('uniform', 'gaussian', 'bimodal')
  - `corruption_rate` (float): Rate of feature corruption (0.0 to 1.0)
  - `uniform_eps` (float): Epsilon for uniform distribution bounds

#### Methods

- `fit(dataset)`: Fit the knowledge component to the dataset
- `transform(x, y=None, **kwargs)`: Apply SCARF corruption to input tensor

### MultiConditionalContext

Generates multiple augmented versions of input data for contrastive learning.

#### Parameters

- `knowledge` (BaseKnowledge): Knowledge component (e.g., SCARFKnowledge)
- `num_augmentations` (int): Number of augmented versions to generate

#### Methods

- `fit(dataset)`: Fit the context to the dataset
- `transform(x, y=None, **kwargs)`: Generate multiple augmented versions
- `transform_single(x, y=None, **kwargs)`: Generate a single augmented version

### SCARFTransform

Registry-compatible wrapper for integration with feature pipelines.

#### Parameters

- `distribution` (str): Distribution type
- `corruption_rate` (float): Corruption rate
- `uniform_eps` (float): Uniform distribution epsilon
- `num_augmentations` (int): Number of augmentations

#### Methods

- `fit(X)`: Fit the transform to DataFrame
- `transform(X)`: Transform DataFrame (returns original for compatibility)
- `transform_tensor(x, y=None)`: Transform tensor data

## Distribution Types

### Uniform Distribution
- **Description**: Samples from uniform distribution between feature bounds
- **Use Case**: When features have clear bounds and uniform distribution is appropriate
- **Parameters**: `uniform_eps` for boundary adjustment

### Gaussian Distribution
- **Description**: Samples from normal distribution with mean and std based on feature bounds
- **Use Case**: When features follow approximately normal distributions
- **Parameters**: Mean = (high + low) / 2, Std = (high - low) / 4

### Bimodal Distribution
- **Description**: Samples from two normal distributions at feature bounds
- **Use Case**: When features have multiple modes or clusters
- **Parameters**: Two normal distributions at low and high bounds

## Complete Examples

### Self-Supervised Learning Pipeline

```python
import torch
import torch.nn as nn
from contexts.multi_conditional_context import MultiConditionalContext, SCARFKnowledge

# 1. Prepare data
data = pd.read_csv('sample_data.csv')
features = data.drop('target', axis=1)

# 2. Create SCARF context
config = {'distribution': 'uniform', 'corruption_rate': 0.4}
knowledge = SCARFKnowledge(config)
context = MultiConditionalContext(knowledge, num_augmentations=2)
context.fit(features)

# 3. Training loop
for epoch in range(num_epochs):
    for batch_x, batch_y in dataloader:
        # Generate positive pairs
        x_augmented, _ = context.transform(batch_x)
        
        # x_augmented shape: (batch_size, 2, num_features)
        # Each sample has 2 augmented versions for contrastive learning
        
        # Pass through encoder
        z1 = encoder(x_augmented[:, 0, :])  # First augmentation
        z2 = encoder(x_augmented[:, 1, :])  # Second augmentation
        
        # Compute contrastive loss
        loss = contrastive_loss(z1, z2)
        
        # Backward pass
        loss.backward()
        optimizer.step()
```

### Batch Processing with CLI

Create a script `process_datasets.sh`:

```bash
#!/bin/bash

# Process training data
python cli_scarf.py --input train.csv --output train_aug.csv \
  --distribution uniform --corruption-rate 0.3 --num-augmentations 1 \
  --target-column target --verbose

# Process validation data
python cli_scarf.py --input val.csv --output val_aug.csv \
  --distribution uniform --corruption-rate 0.3 --num-augmentations 1 \
  --target-column target --verbose

# Process test data (no augmentation)
python cli_scarf.py --input test.csv --output test_aug.csv \
  --corruption-rate 0.0 --num-augmentations 1 \
  --target-column target --verbose

echo "All datasets processed successfully!"
```

### Integration with Feature Pipeline

```python
from feature_transforms.pipeline import ColumnPipeline
from utils.registry import get_transform

# Create pipeline with SCARF
pipeline = ColumnPipeline(
    numeric=['standardize', 'scarf'],
    categorical=['one_hot']
)

# Load data
data = pd.read_csv('sample_data.csv')

# Apply pipeline
X_processed = pipeline.fit_transform(data)
```

## Configuration Examples

### Configuration File (scarf_config.json)

```json
{
  "distribution": "gaussian",
  "corruption_rate": 0.4,
  "uniform_eps": 1e-6
}
```

### Different Use Cases

#### Conservative Augmentation (0.1-0.3)
- **Use Case**: Conservative augmentation, preserving structure
- **Effect**: Subtle changes, maintains original patterns
- **Risk**: May not provide enough diversity

#### Balanced Augmentation (0.3-0.6)
- **Use Case**: Balanced augmentation, general purpose
- **Effect**: Good balance between diversity and information preservation
- **Recommendation**: Start with 0.4-0.5

#### Aggressive Augmentation (0.7-0.9)
- **Use Case**: Strong regularization, aggressive augmentation
- **Effect**: More diverse augmented samples
- **Risk**: May lose too much original information

## Testing

### Run Quick Test
```bash
python quick_test.py
```

### Run Comprehensive Test Suite
```bash
python test_scarf.py
```

### Run Detailed Examples
```bash
python scarf_example.py
```

### Test CLI with Sample Data
```bash
python cli_scarf.py --input sample_data.csv --output test_output.csv --verbose
```

## Troubleshooting

### Common Issues

1. **Import Errors**:
   ```bash
   # Ensure you're in the correct directory
   cd Contextures/contexts/testing_multi_context
   python quick_test.py
   ```

2. **Memory Issues**:
   ```bash
   # Reduce number of augmentations
   python cli_scarf.py --input large_data.csv --output output.csv \
     --num-augmentations 1
   ```

3. **Reproducibility Issues**:
   ```bash
   # Set consistent random seed
   python cli_scarf.py --input data.csv --output output.csv --seed 42
   ```

### Debug Mode

Enable verbose output to see detailed information:
```bash
python cli_scarf.py --input data.csv --output output.csv --verbose
```

Check corruption statistics in Python:
```python
import torch
import pandas as pd

# Load data
data = pd.read_csv('data.csv')
original = torch.tensor(data.drop('target', axis=1).values)

# Apply SCARF
from contexts.multi_conditional_context import SCARFKnowledge
knowledge = SCARFKnowledge({'corruption_rate': 0.3})
knowledge.fit(data.drop('target', axis=1))
corrupted, _ = knowledge.transform(original)

# Check statistics
corruption_diff = torch.abs(original - corrupted)
actual_rate = (corruption_diff > 1e-6).float().mean()
print(f"Actual corruption rate: {actual_rate:.3f}")
```

## Performance Considerations

### Memory Usage
- **Single augmentation**: Minimal memory overhead
- **Multiple augmentations**: Memory scales linearly with number of augmentations
- **Large datasets**: Consider processing in batches

### Computational Cost
- **Uniform distribution**: Fastest
- **Gaussian distribution**: Moderate overhead
- **Bimodal distribution**: Highest overhead due to two distributions

### GPU Usage
- All operations are PyTorch tensor-based
- Automatically uses GPU if available
- Memory efficient for large-scale processing

## Files in this Directory

- **`quick_test.py`** - Simple test script (run with `python quick_test.py`)
- **`test_scarf.py`** - Comprehensive test suite
- **`scarf_example.py`** - Detailed examples with different scenarios
- **`cli_scarf.py`** - Command-line interface for batch processing
- **`sample_data.csv`** - Sample data for testing
- **`README.md`** - This comprehensive documentation

## References

- Original SCARF Paper: [SCARF: Self-Supervised Contrastive Learning using Random Feature Corruption](https://arxiv.org/abs/2106.15147)
- Contextures Framework: This implementation is part of the Contextures package
- PyTorch Distributions: Used for sampling from various distributions

## License

This implementation is part of the Contextures package and follows the same license terms. 