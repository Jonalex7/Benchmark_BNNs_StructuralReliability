# --- Defaults ---

reliability_config_dict = {
    "limit_state": 'g2d_himmelblau',
    "mcs_samples": 1e6,
    "active_samples": 1,
    "active_epochs": 10
}

model_config_dict = {
    "burn_in": 20,
    "sim_steps": 2,
    "N_saved_models": 10,
    "passive_samples": 20,
    "training_epochs": 100,
    "network_architecture": (20, 2),
    "lr": 1e-2,
    "batch_size": 34,
    "verbose": 0,
    "split_train_test": 1.0,
    "seed": None
}