import wandb
import os
import pickle
import yaml
from copy import deepcopy
import sys
import torch
import numpy as np
from datetime import datetime

from methods import REGISTRY as met_REGISTRY
from utils.data import get_dataloader, normalize_data, denormalize_data

def training_loop(config=None):
    
    with wandb.init(config=config):
        config = wandb.config
        patience = 50
        split_train_test = 0.9
        verbose = None
        training_epochs = 5000
        set_ = 6
        replication = 0
        results_file = {}

        model_bnn = met_REGISTRY[config.model]

        #loading default_bench arguments
        batch_size = config.batch_size
        learning_rate = config.learning_rate
        hidden_sizes = config.hidden_size
        hidden_layers = config.layers
        
        w_decay = config.weight_decay
        num_forwards = config.num_forwards
        seed = config.seed
        valid_samples = config.validation_size

        # Seed definition
        if seed is not None:
            seed_experiment = seed
        else:
            seed_experiment = np.random.randint(0, 2**30 - 1)
        np.random.seed(seed_experiment)
        torch.manual_seed(seed_experiment)

        args = {}

        dropout_probability = config.dropout
        args['dropout_probability'] = dropout_probability

        # kl_scale = config.kl_scale
        # args['kl_scale'] = kl_scale

        args['n_simulations'] = num_forwards
        args['weight_decay'] = w_decay

        file_name = config.dataset

        with open(
            os.path.join(
                os.path.dirname(__file__),
                "bench_cases",
                "data",
                "{}.pkl".format(file_name),
            ),
            "rb",
        ) as f:
            try:
                data = pickle.load(f)
            except Exception as e:
                print("Error loading .pkl file:", e)

        # Directory to save results
        results_dir = 'bench_cases/results/'+'hyper_tune_'+ file_name +'/'+ config.model
        date_time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        x_train = data['training'][set_][replication][1]
        y_train = data['training'][set_][replication][2]

        x_valid = data['validation'][1][:valid_samples] # idx 0 has normalised inputs (0,1)
        y_valid = data['validation'][2][:valid_samples] # dependent variable, shape (100_000,)

        print('Data set: ', file_name, '-----------------------------------------------------')
        print('Training set size: ', x_train.shape[0], end=" ")
        print('Replication: ', replication+1)

        input_dim = x_train.shape[1]
        output_dim = y_train.shape[1]

        x_norm, x_mean, x_var = normalize_data(x_train)
        y_norm, y_mean, y_var = normalize_data(y_train)

        model = model_bnn(input_size=input_dim, hidden_sizes=hidden_sizes, 
                    hidden_layers=hidden_layers, output_size=output_dim, 
                    args=args)
        
        train_loader, test_loader = get_dataloader(x_norm, y_norm, split_train_test, batch_size)
       
        train_loss, valid_loss = model.train(train_loader, test_loader, num_epochs=training_epochs, 
                                            lr=learning_rate, patience=patience, verbose=verbose)

        
        wandb.log({"valid_loss": valid_loss})
        wandb.log({"train_loss": train_loss})

        # Validation error - RSUQ metric, Relative MSE
        print('Estimating Rel. MSE...')
        
        y_random = torch.empty(len(x_valid), num_forwards)
        with torch.no_grad():
            x_valid_norm = normalize_data(x_valid, mean=x_mean, variance=x_var)
            for i in range(num_forwards):
                prediction = model(x_valid_norm).reshape(-1)
                y_random[:, i] = denormalize_data(prediction, mean=y_mean, variance=y_var)

        y_mean = y_random.mean(dim=1).reshape(-1,1)
        # y_mean_val= denormalize_data(y_mean_norm, mean=y_mean, variance=y_var)

        rel_mse = model.criterion(y_valid, y_mean)/(torch.var(y_valid))

        wandb.log({"rela_mse": rel_mse})

        results_file['seed'] = seed_experiment
        results_file['config'] = config_dict
        results_file['metrics'] = train_loss, valid_loss, rel_mse
        results_file['model'] = model
        results_file['x_mean_var'] = x_mean, x_var
        results_file['y_mean_var'] = y_mean, y_var

        #Saving results file
        with open(results_dir +'/'+'set_'+str(set_)+'_rep_'+str(replication) + "_" + date_time_stamp + ".pkl", 'wb') as file_id:
            pickle.dump(results_file, file_id)

if __name__ == "__main__":
    params = deepcopy(sys.argv)
    # Get the defaults from default_hypertune.yaml
    with open(
        os.path.join(os.path.dirname(__file__), "config", "default_hypertune.yaml"), "r"
    ) as f:
        try:
            config_dict = yaml.full_load(f)
        except yaml.YAMLError as exc:
            assert False, "default_hypertune.yaml error: {}".format(exc)

    # Initialize the sweep
    sweep_id = wandb.sweep(config_dict, project="bench_rsuq_dropout_Borehole")

    # Run the sweep
    wandb.agent(sweep_id, function=training_loop, count=10)