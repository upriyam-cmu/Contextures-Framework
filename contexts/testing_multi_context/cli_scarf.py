#!/usr/bin/env python3
"""
SCARF Command Line Interface

A command-line tool for running SCARF (Self-supervised Contrastive Learning using Random Feature Corruption)
augmentation on tabular data.
"""

import argparse
import sys
import os
import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any

# Add the current directory to the path to import Contextures
sys.path.append(str(Path(__file__).parent.parent.parent))

from contexts.multi_conditional_context import (
    SCARFKnowledge, 
    MultiConditionalContext, 
    SCARFTransform
)
from utils.registry import get_transform


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from various file formats."""
    file_path = Path(file_path)
    
    if file_path.suffix.lower() == '.csv':
        return pd.read_csv(file_path)
    elif file_path.suffix.lower() in ['.xlsx', '.xls']:
        return pd.read_excel(file_path)
    elif file_path.suffix.lower() == '.parquet':
        return pd.read_parquet(file_path)
    elif file_path.suffix.lower() == '.json':
        return pd.read_json(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def save_data(data: pd.DataFrame, file_path: str) -> None:
    """Save data to various file formats."""
    file_path = Path(file_path)
    
    if file_path.suffix.lower() == '.csv':
        data.to_csv(file_path, index=False)
    elif file_path.suffix.lower() in ['.xlsx', '.xls']:
        data.to_excel(file_path, index=False)
    elif file_path.suffix.lower() == '.parquet':
        data.to_parquet(file_path, index=False)
    elif file_path.suffix.lower() == '.json':
        data.to_json(file_path, orient='records')
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def create_scarf_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Create SCARF configuration from command line arguments."""
    config = {
        'distribution': args.distribution,
        'corruption_rate': args.corruption_rate,
        'uniform_eps': args.uniform_eps,
    }
    return config


def run_scarf_augmentation(
    data: pd.DataFrame,
    config: Dict[str, Any],
    num_augmentations: int = 1,
    target_column: Optional[str] = None,
    output_format: str = 'tensor'
) -> Any:
    """
    Run SCARF augmentation on the data.
    
    Args:
        data: Input DataFrame
        config: SCARF configuration
        num_augmentations: Number of augmented versions to generate
        target_column: Optional target column to exclude from augmentation
        output_format: Output format ('tensor', 'dataframe', 'numpy')
    
    Returns:
        Augmented data in the specified format
    """
    # Separate features and target
    if target_column and target_column in data.columns:
        features = data.drop(columns=[target_column])
        target = data[target_column]
    else:
        features = data
        target = None
    
    # Create SCARF knowledge
    knowledge = SCARFKnowledge(config)
    knowledge.fit(features)
    
    # Create multi-conditional context
    context = MultiConditionalContext(knowledge, num_augmentations)
    context.fit(features)
    
    # Convert to tensor
    x_tensor = torch.tensor(features.values, dtype=torch.float32)
    y_tensor = torch.tensor(target.values, dtype=torch.float32) if target is not None else None
    
    # Apply augmentation
    x_augmented, y = context.transform(x_tensor, y_tensor)
    
    # Convert to desired output format
    if output_format == 'tensor':
        return x_augmented, y
    elif output_format == 'numpy':
        return x_augmented.numpy(), y.numpy() if y is not None else None
    elif output_format == 'dataframe':
        # For DataFrame output, we'll create multiple DataFrames or a single one
        if num_augmentations == 1:
            # Single augmentation
            augmented_df = pd.DataFrame(
                x_augmented.squeeze(1).numpy(),
                columns=features.columns,
                index=features.index
            )
            if target is not None:
                augmented_df[target_column] = target
            return augmented_df
        else:
            # Multiple augmentations - return list of DataFrames
            augmented_dfs = []
            for i in range(num_augmentations):
                aug_df = pd.DataFrame(
                    x_augmented[:, i, :].numpy(),
                    columns=features.columns,
                    index=features.index
                )
                if target is not None:
                    aug_df[target_column] = target
                augmented_dfs.append(aug_df)
            return augmented_dfs
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description='SCARF (Self-supervised Contrastive Learning using Random Feature Corruption) CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  python cli_scarf.py --input data.csv --output augmented_data.csv

  # Multiple augmentations with custom settings
  python cli_scarf.py --input data.csv --output augmented_data.csv \\
    --distribution gaussian --corruption-rate 0.4 --num-augmentations 3

  # Save as tensor format
  python cli_scarf.py --input data.csv --output augmented_data.pt \\
    --output-format tensor --num-augmentations 2

  # Exclude target column from augmentation
  python cli_scarf.py --input data.csv --output augmented_data.csv \\
    --target-column target --corruption-rate 0.3

  # Use bimodal distribution
  python cli_scarf.py --input data.csv --output augmented_data.csv \\
    --distribution bimodal --corruption-rate 0.5
        """
    )
    
    # Input/Output arguments
    parser.add_argument('--input', '-i', required=True,
                       help='Input data file (CSV, Excel, Parquet, JSON)')
    parser.add_argument('--output', '-o', required=True,
                       help='Output file path')
    parser.add_argument('--output-format', choices=['tensor', 'dataframe', 'numpy'],
                       default='dataframe', help='Output format')
    
    # SCARF configuration
    parser.add_argument('--distribution', choices=['uniform', 'gaussian', 'bimodal'],
                       default='uniform', help='Distribution type for corruption')
    parser.add_argument('--corruption-rate', type=float, default=0.5,
                       help='Rate of feature corruption (0.0 to 1.0)')
    parser.add_argument('--uniform-eps', type=float, default=1e-6,
                       help='Epsilon for uniform distribution bounds')
    parser.add_argument('--num-augmentations', type=int, default=1,
                       help='Number of augmented versions to generate')
    
    # Data processing
    parser.add_argument('--target-column', type=str,
                       help='Target column to exclude from augmentation')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # Additional options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--save-config', type=str,
                       help='Save configuration to JSON file')
    parser.add_argument('--load-config', type=str,
                       help='Load configuration from JSON file')
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    try:
        # Load configuration if specified
        if args.load_config:
            with open(args.load_config, 'r') as f:
                config = json.load(f)
            if args.verbose:
                print(f"Loaded configuration from {args.load_config}")
        else:
            config = create_scarf_config(args)
        
        # Save configuration if specified
        if args.save_config:
            with open(args.save_config, 'w') as f:
                json.dump(config, f, indent=2)
            if args.verbose:
                print(f"Saved configuration to {args.save_config}")
        
        # Load data
        if args.verbose:
            print(f"Loading data from {args.input}")
        data = load_data(args.input)
        
        if args.verbose:
            print(f"Data shape: {data.shape}")
            print(f"Columns: {list(data.columns)}")
            if args.target_column:
                print(f"Target column: {args.target_column}")
        
        # Run SCARF augmentation
        if args.verbose:
            print(f"Running SCARF augmentation with {args.num_augmentations} augmentations")
            print(f"Configuration: {config}")
        
        result = run_scarf_augmentation(
            data=data,
            config=config,
            num_augmentations=args.num_augmentations,
            target_column=args.target_column,
            output_format=args.output_format
        )
        
        # Save results
        if args.output_format == 'tensor':
            # Save as PyTorch tensor
            torch.save(result, args.output)
            if args.verbose:
                print(f"Saved tensor to {args.output}")
                print(f"Tensor shape: {result[0].shape if isinstance(result, tuple) else result.shape}")
        elif args.output_format == 'numpy':
            # Save as numpy arrays
            if isinstance(result, tuple):
                np.savez(args.output, x=result[0], y=result[1])
            else:
                np.save(args.output, result)
            if args.verbose:
                print(f"Saved numpy arrays to {args.output}")
        elif args.output_format == 'dataframe':
            # Save as DataFrame(s)
            if isinstance(result, list):
                # Multiple augmentations
                base_path = Path(args.output)
                for i, df in enumerate(result):
                    output_path = base_path.parent / f"{base_path.stem}_aug_{i+1}{base_path.suffix}"
                    save_data(df, str(output_path))
                    if args.verbose:
                        print(f"Saved augmentation {i+1} to {output_path}")
            else:
                # Single augmentation
                save_data(result, args.output)
                if args.verbose:
                    print(f"Saved augmented data to {args.output}")
        
        if args.verbose:
            print("SCARF augmentation completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main() 