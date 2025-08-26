#!/usr/bin/env python3
"""
Test script to verify LinOSS setup and dataset creation.
"""

import os
import sys
import jax
import jax.numpy as jnp

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import equinox as eqx
        print("✓ Equinox imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Equinox: {e}")
        return False
    
    try:
        import optax
        print("✓ Optax imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Optax: {e}")
        return False
    
    try:
        import diffrax
        print("✓ Diffrax imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Diffrax: {e}")
        return False
    
    try:
        import signax
        print("✓ Signax imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Signax: {e}")
        return False
    
    return True

def test_jax():
    """Test if JAX is working correctly."""
    print("\nTesting JAX...")
    
    try:
        # Test basic JAX operations
        x = jnp.array([1, 2, 3])
        y = jnp.array([4, 5, 6])
        z = x + y
        print(f"✓ JAX basic operations work: {z}")
        
        # Test JAX device
        print(f"✓ JAX device: {jax.devices()}")
        
        return True
    except Exception as e:
        print(f"✗ JAX test failed: {e}")
        return False

def test_dataset_creation():
    """Test if we can create the synthetic dataset."""
    print("\nTesting dataset creation...")
    
    try:
        from data_dir.create_simple_dataset import create_synthetic_dataset
        
        # Create a small test dataset
        dataset = create_synthetic_dataset(
            n_samples=100,  # Small for testing
            n_timesteps=50,
            n_features=2,
            n_classes=2
        )
        
        print(f"✓ Dataset created successfully")
        print(f"  - Train samples: {dataset['train'][0].shape}")
        print(f"  - Val samples: {dataset['val'][0].shape}")
        print(f"  - Test samples: {dataset['test'][0].shape}")
        print(f"  - Data dimension: {dataset['data_dim']}")
        print(f"  - Label dimension: {dataset['label_dim']}")
        
        return True
    except Exception as e:
        print(f"✗ Dataset creation failed: {e}")
        return False

def test_model_import():
    """Test if we can import the LinOSS model."""
    print("\nTesting model import...")
    
    try:
        from models.LinOSS import LinOSS
        print("✓ LinOSS model imported successfully")
        return True
    except Exception as e:
        print(f"✗ Model import failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("LinOSS Setup Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_jax,
        test_dataset_creation,
        test_model_import
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! LinOSS is ready to use.")
        print("\nNext steps:")
        print("1. Generate dataset: python data_dir/create_simple_dataset.py")
        print("2. Train LinOSS: python run_experiment.py --model_names LinOSS --dataset_names simple_oscillatory --experiment_folder experiment_configs/repeats")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    print("=" * 50)

if __name__ == "__main__":
    main()

