#!/usr/bin/env python3
"""
Quick test script for SCARF functionality.
Run this with: python quick_test.py
"""

import sys
import torch
import pandas as pd
from pathlib import Path

# Add the parent directory to the path to import Contextures
sys.path.append(str(Path(__file__).parent.parent.parent))

def quick_test():
    """Quick test of SCARF functionality."""
    print("🧪 Quick SCARF Test")
    print("=" * 40)
    
    try:
        # Import SCARF components
        from contexts.multi_conditional_context import SCARFKnowledge, MultiConditionalContext
        print("✅ Imports successful")
        
        # Create sample data
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50],
            'target': [0, 1, 0, 1, 0]
        })
        print(f"✅ Sample data created: {data.shape}")
        
        # Test SCARF knowledge
        config = {
            'distribution': 'uniform',
            'corruption_rate': 0.3,
            'uniform_eps': 1e-6
        }
        
        knowledge = SCARFKnowledge(config)
        knowledge.fit(data.drop('target', axis=1))
        print("✅ SCARF knowledge fitted")
        
        # Test transformation
        x_tensor = torch.tensor(data.drop('target', axis=1).values, dtype=torch.float32)
        x_corrupted, y = knowledge.transform(x_tensor)
        print(f"✅ Transformation successful: {x_tensor.shape} -> {x_corrupted.shape}")
        
        # Test multi-conditional context
        context = MultiConditionalContext(knowledge, num_augmentations=2)
        context.fit(data.drop('target', axis=1))
        x_augmented, _ = context.transform(x_tensor)
        print(f"✅ Multi-augmentation successful: {x_augmented.shape}")
        
        # Calculate corruption statistics
        corruption_diff = torch.abs(x_tensor - x_corrupted)
        actual_rate = (corruption_diff > 1e-6).float().mean()
        print(f"✅ Actual corruption rate: {actual_rate:.3f}")
        
        print("\n🎉 All tests passed! SCARF is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = quick_test()
    sys.exit(0 if success else 1) 