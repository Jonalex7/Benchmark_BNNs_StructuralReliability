# --- Defaults ---

reliability_config_dict = {
    "limit_state": 'g11d_electric',
    "mcs_samples": 1e6,
    "active_samples": 5,
    "active_epochs": 10
}

model_config_dict = {
    "KL_scale": 50,
    "n_simulations": 100,
    "passive_samples": 100,
    "training_epochs": 2000,
    "network_architecture": (30, 2),
    "lr": 1e-2,
    "batch_size": 64,
    "verbose": 0,
    "split_train_test": 1.0,
    "seed": None
}