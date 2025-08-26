"""
Test script for your trained LinOSS model on light curves dataset.
This script loads the trained model and evaluates it on test data.
"""

import os
import pickle
import jax.numpy as jnp
from train import create_dataset_model_and_train

def test_light_curves():
    """Test the trained LinOSS model on your light curves dataset."""
    
    # Load your processed dataset
    dataset_path = "data_dir/processed/light_curves"
    
    # Load data and labels
    with open(os.path.join(dataset_path, 'data.pkl'), 'rb') as f:
        data = pickle.load(f)
    
    with open(os.path.join(dataset_path, 'labels.pkl'), 'rb') as f:
        labels = pickle.load(f)
    
    with open(os.path.join(dataset_path, 'original_idxs.pkl'), 'rb') as f:
        train_idxs, val_idxs, test_idxs = pickle.load(f)
    
    print(f"📊 Dataset loaded:")
    print(f"   Total samples: {data.shape[0]}")
    print(f"   Time steps: {data.shape[1]}")
    print(f"   Features: {data.shape[2]}")
    print(f"   Train samples: {len(train_idxs)}")
    print(f"   Val samples: {len(val_idxs)}")
    print(f"   Test samples: {len(test_idxs)}")
    
    # Test configuration (same as training but for evaluation)
    model_args = {
        "num_blocks": 2,
        "hidden_dim": 64,
        "ssm_dim": 32,
        "ssm_blocks": 2,
        "dt0": None,
        "solver": None,
        "stepsize_controller": None,
        "scale": 0,
        "lambd": None,
    }
    
    test_args = {
        "data_dir": "data_dir",
        "use_presplit": True,
        "dataset_name": "light_curves",
        "output_step": 1,
        "metric": "accuracy",
        "include_time": True,
        "T": 1,
        "model_name": "LinOSS",
        "stepsize": 1,
        "logsig_depth": 1,
        "linoss_discretization": "IM",
        "model_args": model_args,
        "num_steps": 100,  # Fewer steps for testing
        "print_steps": 50,
        "lr": 0.001,
        "lr_scheduler": lambda lr: lr,
        "batch_size": 32,
        "output_parent_dir": "outputs",
        "id": None,
    }
    
    print("\n🧪 Testing model...")
    
    # Test with a single seed for evaluation
    test_seed = 42
    create_dataset_model_and_train(seed=test_seed, **test_args)
    
    print("\n✅ Testing completed!")
    print("📁 Check the outputs/ directory for results")

if __name__ == "__main__":
    test_light_curves()
