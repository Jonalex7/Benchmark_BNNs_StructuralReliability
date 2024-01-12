# --- Defaults ---

reliability_config_dict = {
    "limit_state": 'g2d_himmelblau',
    "mcs_samples": 1e6,
    "active_samples": 1,
    "active_epochs": 10
}

model_config_dict = {
    "n_ensembles": 5,
    "passive_samples": 20,
    "training_epochs": 1000,
    "network_architecture": (50, 2),
    "lr": 1e-2,
    "batch_size": 34,
    "verbose": 0,
    "split_train_test": 1.0,
    "seed": None
}