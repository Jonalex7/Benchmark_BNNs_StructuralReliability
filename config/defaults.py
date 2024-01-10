# --- Defaults ---

reliability_config_dict = {
    "limit_state": 'g11d_electric',
    "mcs_samples": 1e6,
    "active_points": 5,
    "active_epochs": 10
}

model_config_dict = {
    "passive_samples": 100,
    "training_epochs": 100,
    "network_architecture": (2, 20, 2, 1),
    "lr": 1e-2,
    "batch_size": 64
    "seed": None
}



# # --- Model configuration ---
# passive_samples: 10 # Initial samples for DoE
# training_epochs: 1000 # Number of training epochs 
# network_architecture: (2, 100, 2, 1) # Input, width, layers, output sizes
# lr: 0.1 # Learning rate
# batch_size: 32 # Batch size for each epoch
# active_points: 5 # Number of new labelled points

# # --- Reliability settings ---
# mcs_samples: int(1e6) # Size of Monte carlo population