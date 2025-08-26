"""
Template script to load your own dataset for LinOSS training.
Modify this script to load your specific data format.
"""

import os
import pickle
import jax.numpy as jnp
import torch

def load_your_data():
    """
    Load your own dataset from multiple PyTorch .pt files containing labels, flux, and time.
    
    Modify the file paths below to match your actual .pt files.
    """
    labels_file = "qlp_train_tensors/labels_train_qlp.pt"
    flux_file = "qlp_train_tensors/flux_train_qlp.pt"
    time_file = "qlp_train_tensors/time_train_qlp.pt"
    
    # Load PyTorch files
    labels = torch.load(labels_file, map_location='cpu')
    flux_data = torch.load(flux_file, map_location='cpu')
    time_data = torch.load(time_file, map_location='cpu')
    
    print("Loaded all PyTorch .pt files successfully")
    
    # Keep as PyTorch tensors - no numpy conversion needed
    # Stack time and flux along the last dimension: (n_stars, n_timesteps, 2)
    data = torch.stack([time_data, flux_data], dim=-1)
    
    # Determine number of classes from labels
    n_classes = labels.shape[1] if len(labels.shape) > 1 else 1
    print(f"Number of classes: {n_classes}")
    print(f"Final data shape: {data.shape}")
    
    return data, labels

def create_train_val_test_splits(data, labels, train_ratio=0.7, val_ratio=0.15):
    """Create train/validation/test splits from your data."""
    n_samples = data.shape[0]  # Use shape[0] for PyTorch tensors
    
    # Calculate split sizes
    n_train = int(train_ratio * n_samples)
    n_val = int(val_ratio * n_samples)
    n_test = n_samples - n_train - n_val
    
    # Create splits using PyTorch indexing
    train_data = data[:n_train]
    train_labels = labels[:n_train]
    
    val_data = data[n_train:n_train+n_val]
    val_labels = labels[n_train:n_train+n_val]
    
    test_data = data[n_train+n_val:]
    test_labels = labels[n_train+n_val:]
    
    return {
        'train': (train_data, train_labels),
        'val': (val_data, val_labels),
        'test': (test_data, test_labels)
    }

def save_dataset_for_linoss(dataset, save_dir, dataset_name):
    """Save the dataset in the format required by LinOSS."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract data and labels
    train_data, train_labels = dataset['train']
    val_data, val_labels = dataset['val']
    test_data, test_labels = dataset['test']
    
    # Concatenate all data (PyTorch tensors)
    all_data = torch.cat([train_data, val_data, test_data], dim=0)
    all_labels = torch.cat([train_labels, val_labels, test_labels], dim=0)
    
    # Convert to JAX arrays for LinOSS
    all_data = jnp.array(all_data.numpy())
    all_labels = jnp.array(all_labels.numpy())
    
    # Save data and labels
    with open(os.path.join(save_dir, 'data.pkl'), 'wb') as f:
        pickle.dump(all_data, f)
    
    with open(os.path.join(save_dir, 'labels.pkl'), 'wb') as f:
        pickle.dump(all_labels, f)
    
    # Create and save train/val/test indices
    n_train = len(train_data)
    n_val = len(val_data)
    n_test = len(test_data)
    
    train_idxs = jnp.arange(n_train)
    val_idxs = jnp.arange(n_train, n_train + n_val)
    test_idxs = jnp.arange(n_train + n_val, n_train + n_val + n_test)
    
    original_idxs = [train_idxs, val_idxs, test_idxs]
    
    with open(os.path.join(save_dir, 'original_idxs.pkl'), 'wb') as f:
        pickle.dump(original_idxs, f)
    
    print(f"Dataset '{dataset_name}' saved successfully!")
    print(f"Location: {save_dir}")
    print(f"Data shape: {all_data.shape}")
    print(f"Labels shape: {all_labels.shape}")
    print(f"Train samples: {n_train}")
    print(f"Val samples: {n_val}")
    print(f"Test samples: {n_test}")

def main():
    """Main function to load and save your dataset."""
    print("Loading your dataset...")
    
    # Step 1: Load your data
    data, labels = load_your_data()
    
    # Labels are already one-hot encoded - no conversion needed
    
    # Step 2: Create train/val/test splits
    dataset = create_train_val_test_splits(data, labels)
    
    # Step 3: Save in LinOSS format
    dataset_name = "light_curves"  # Change this to your dataset name
    save_dir = f"data_dir/processed/{dataset_name}"
    
    save_dataset_for_linoss(dataset, save_dir, dataset_name)

if __name__ == "__main__":
    main()
