"""
Simplified script to train LinOSS on your light curves dataset.
This script bypasses the complex experiment framework and directly trains the model.
"""

import os
import json
from train import create_dataset_model_and_train

def train_light_curves():
    """Train LinOSS on your light curves dataset with hardcoded parameters."""
    
    # Load your config
    config_path = "experiment_configs/repeats/LinOSS/light_curves.json"
    
    with open(config_path, "r") as file:
        config = json.load(file)
    
    print("Starting LinOSS training on your light curves dataset...")
    
    # Model arguments for LinOSS
    model_args = {
        "num_blocks": int(config["num_blocks"]),
        "hidden_dim": int(config["hidden_dim"]),
        "ssm_dim": int(config["ssm_dim"]),
        "ssm_blocks": int(config["ssm_blocks"]),
        "dt0": None,
        "solver": None,  # Will use default in LinOSS
        "stepsize_controller": None,  # Will use default in LinOSS
        "scale": config["scale"],
        "lambd": None,
    }
    
    # Training arguments
    run_args = {
        "data_dir": config["data_dir"],
        "use_presplit": config["use_presplit"],
        "dataset_name": config["dataset_name"],
        "output_step": 1,
        "metric": config["metric"],
        "include_time": config["time"].lower() == "true",
        "T": config["T"],
        "model_name": config["model_name"],
        "stepsize": 1,
        "logsig_depth": 1,
        "linoss_discretization": config["linoss_discretization"],
        "model_args": model_args,
        "num_steps": config["num_steps"],
        "print_steps": config["print_steps"],
        "lr": float(config["lr"]),
        "lr_scheduler": eval(config["lr_scheduler"]),
        "batch_size": config["batch_size"],
        "output_parent_dir": config["output_parent_dir"],
        "id": None,
    }
    
    # Train with each seed
    for seed in config["seeds"]:
        print(f"\n🌱 Training with seed: {seed}")
        print("=" * 50)
        
        create_dataset_model_and_train(seed=seed, **run_args)

if __name__ == "__main__":
    train_light_curves()
