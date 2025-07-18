#!/usr/bin/env python3
"""
Simple test script for SCARF functionality.
"""

import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path

# Add the current directory to the path
sys.path.append(str(Path(__file__).parent.parent.parent))

def test_imports():
    """Test that all SCARF components can be imported."""
    print("Testing imports...")
    
    try:
        from contexts.multi_conditional_context import (
            SCARFKnowledge, 
            MultiConditionalContext, 
            SCARFTransform
        )
        from utils.registry import get_transform
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_basic_functionality():
    """Test basic SCARF functionality."""
    print("\nTesting basic functionality...")
    
    try:
        from contexts.multi_conditional_context import SCARFKnowledge
        
        # Create sample data
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50],
            'target': [0, 1, 0, 1, 0]
        })
        
        # Test SCARF knowledge
        config = {
            'distribution': 'uniform',
            'corruption_rate': 0.3,
            'uniform_eps': 1e-6
        }
        
        knowledge = SCARFKnowledge(config)
        knowledge.fit(data.drop('target', axis=1))
        
        # Test transformation
        x_tensor = torch.tensor(data.drop('target', axis=1).values, dtype=torch.float32)
        x_corrupted, y = knowledge.transform(x_tensor)
        
        print(f"✓ Original shape: {x_tensor.shape}")
        print(f"✓ Corrupted shape: {x_corrupted.shape}")
        print(f"✓ Basic functionality works")
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality error: {e}")
        return False


def test_multi_conditional_context():
    """Test multi-conditional context."""
    print("\nTesting multi-conditional context...")
    
    try:
        from contexts.multi_conditional_context import SCARFKnowledge, MultiConditionalContext
        
        # Create sample data
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50],
        })
        
        # Test multi-conditional context
        config = {'distribution': 'gaussian', 'corruption_rate': 0.4}
        knowledge = SCARFKnowledge(config)
        context = MultiConditionalContext(knowledge, num_augmentations=3)
        context.fit(data)
        
        # Test transformation
        x_tensor = torch.tensor(data.values, dtype=torch.float32)
        x_augmented, _ = context.transform(x_tensor)
        
        print(f"✓ Original shape: {x_tensor.shape}")
        print(f"✓ Augmented shape: {x_augmented.shape}")
        print(f"✓ Multi-conditional context works")
        return True
        
    except Exception as e:
        print(f"✗ Multi-conditional context error: {e}")
        return False


def test_registry():
    """Test registry integration."""
    print("\nTesting registry integration...")
    
    try:
        from utils.registry import get_transform
        
        # Test getting SCARF transform from registry
        scarf_transform = get_transform('scarf')(
            distribution='bimodal',
            corruption_rate=0.5,
            num_augmentations=2
        )
        
        print(f"✓ SCARF transform retrieved from registry")
        print(f"✓ Transform type: {type(scarf_transform)}")
        return True
        
    except Exception as e:
        print(f"✗ Registry error: {e}")
        return False


def test_distributions():
    """Test different distribution types."""
    print("\nTesting different distributions...")
    
    try:
        from contexts.multi_conditional_context import SCARFKnowledge
        
        # Create sample data
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50],
        })
        
        distributions = ['uniform', 'gaussian', 'bimodal']
        
        for dist in distributions:
            config = {'distribution': dist, 'corruption_rate': 0.3}
            knowledge = SCARFKnowledge(config)
            knowledge.fit(data)
            
            x_tensor = torch.tensor(data.values, dtype=torch.float32)
            x_corrupted, _ = knowledge.transform(x_tensor)
            
            print(f"✓ {dist} distribution works")
        
        return True
        
    except Exception as e:
        print(f"✗ Distribution test error: {e}")
        return False


def main():
    """Run all tests."""
    print("SCARF Test Suite")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_basic_functionality,
        test_multi_conditional_context,
        test_registry,
        test_distributions
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! SCARF is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 