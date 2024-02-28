import wandb
import os
import pickle
import yaml
from copy import deepcopy
import sys
import collections
import torch

from methods import REGISTRY as met_REGISTRY
from utils.data import get_dataloader, normalize_data, denormalize_data

def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            method_name = config_name
            del params[_i]
            break

    if config_name is not None:
        with open(
            os.path.join(
                os.path.dirname(__file__),
                "config",
                subfolder,
                "{}.yaml".format(config_name),
            ),
            "r",
        ) as f:
            try:
                config_dict = yaml.full_load(f)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict, method_name
    
def _get_data(params, arg_name, subfolder):
    data_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            data_name = _v.split("=")[1]
            file_name = data_name
            del params[_i]
            break

    if data_name is not None:
        with open(
            os.path.join(
                os.path.dirname(__file__),
                "bench_cases",
                subfolder,
                "{}.pkl".format(data_name),
            ),
            "rb",
        ) as f:
            try:
                data_set = pickle.load(f)
            except Exception as e:
                print("Error loading .pkl file:", e)
        return data_set, file_name
    
def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def training_loop(config=None):
    
    with wandb.init(config=config):
        config = wandb.config
        patience = 50
        split_train_test = 0.9
        verbose = None
        training_epochs = 5000
        set_ = 6
        replication = 0

        model_bnn = met_REGISTRY[config.model]

        #loading default_bench arguments
        batch_size = config.batch_size
        learning_rate = config.learning_rate
        hidden_sizes = config.hidden_size
        hidden_layers = config.layers
        dropout_probability = config.dropout
        w_decay = config.weight_decay
        num_forwards = config.num_forwards

        args = {}
        args['dropout_probability'] = dropout_probability
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

        x_train = data['training'][set_][replication][1]
        y_train = data['training'][set_][replication][2]

        x_valid = data['validation'][1] # idx 0 has normalised inputs (0,1)
        y_valid = data['validation'][2] # dependent variable, shape (100_000,)

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

if __name__ == "__main__":
    params = deepcopy(sys.argv)
    # Get the defaults from default.yaml
    with open(
        os.path.join(os.path.dirname(__file__), "config", "default_hypertune.yaml"), "r"
    ) as f:
        try:
            config_dict = yaml.full_load(f)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # Load algorithm and env base configs
    # model_dict, method_name = _get_config(params, "--method", "mod")
    # config_dict = recursive_dict_update(config_dict, model)
            
    # data, file_name = _get_data(params, "--data", "data")

    # Initialize the sweep
    sweep_id = wandb.sweep(config_dict, project="bench_rsuq_Borehole")

    # Run the sweep
    wandb.agent(sweep_id, function=training_loop, count=10)