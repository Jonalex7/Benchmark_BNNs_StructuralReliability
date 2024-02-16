import os
import collections
import numpy as np
import torch
from datetime import datetime
import pickle
import yaml
from copy import deepcopy
from os.path import dirname
import sys

from limit_states import REGISTRY as ls_REGISTRY
from methods import REGISTRY as met_REGISTRY
from utils.data import get_dataloader, isoprob_transform
from active_training.active_train import ActiveTrain

def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
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
        return config_dict
    
def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

if __name__ == "__main__":
    params = deepcopy(sys.argv)
    # Get the defaults from default.yaml
    with open(
        os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r"
    ) as f:
        try:
            config_dict = yaml.full_load(f)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # Load algorithm and env base configs
    model = _get_config(params, "--method", "mod")
    lstate = _get_config(params, "--lstate", "ls")

    config_dict = recursive_dict_update(config_dict, model)
    config_dict = recursive_dict_update(config_dict, lstate)

    # Directory to save results
    results_dir, results_file = config_dict['res_dir'], config_dict['res_file']
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    date_time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    number_exp = config_dict['number_exp']
    passive_samples = config_dict['passive_samples'] 

    # Loading limit state
    lstate = ls_REGISTRY[config_dict['l_state']]()
    act_train = ActiveTrain()
    print('Target limit state: ', config_dict['l_state'])
    mcs_samples = config_dict['mcs_samples']
    eval = config_dict['evaluate']
    pf, beta, _, y_mc_test = lstate.monte_carlo_estimate(mcs_samples)
    y_max = np.max(y_mc_test)   #to normalise the output for training
    print('ref. PF:', pf, 'B:',beta)

    print('Method: ', config_dict['method'])
    # Neural net config.
    method = met_REGISTRY[config_dict['method']]
    hidden_sizes, hidden_layers = config_dict['hidden_sizes'], config_dict['hidden_layers']
    lr, batch_size = config_dict['learning_rate'], config_dict['batch_size']
    split_train_test = config_dict['split_train_test']
    verbose = config_dict['verbose']
    args = config_dict['args']

    use_cuda = torch.cuda.is_available() 

    #required for SGHMC only
    if config_dict['method'] == 'sghmc':
        args['learning_rate'] = lr                  #for the optimizer
        args['passive_samples'] = passive_samples   #to estimate a loss over the whole dataset
        args['cuda'] = use_cuda                     #we could keep it to use cuda in all methods

    # Active training args
    training_epochs = config_dict['training_epochs']
    n_train_ep = config_dict['active_epochs']
    active_points = config_dict['active_samples']

    for exp in range(number_exp):

        # Seed definition
        if config_dict['seed'] is not None:
            seed_experiment = config_dict['seed']
        else:
            seed_experiment = np.random.randint(0, 2**30 - 1)
        np.random.seed(seed_experiment)
        torch.manual_seed(seed_experiment)

        # Passive training
        x_norm, x_scaled, y_scaled = lstate.get_doe_points(n_samples=passive_samples, method='lhs')   
        results_dict = {}

        # Active training loop
        for ep in range(n_train_ep):

            model = method(input_size=lstate.input_dim, hidden_sizes=hidden_sizes, 
                        hidden_layers=hidden_layers, output_size=lstate.output_dim, 
                        args=args)
            
            x_train = torch.tensor(x_norm, dtype=torch.float32)
            # y_train = torch.tensor(y_scaled, dtype=torch.float32).view(-1,1)
            y_train = torch.tensor(y_scaled/y_max, dtype=torch.float32).view(-1, lstate.output_dim)   #normalised output
            
            print('Samples: ', x_train.shape[0], end=" ")
            train_loader, _ = get_dataloader(x_train, y_train, lstate.input_dim, lstate.output_dim, 
                                            split_train_test, batch_size)

            model.train(train_loader, num_epochs=training_epochs, lr=lr, verbose=verbose)

            if eval is True:
                Pf_ref, B_ref, x_mc_norm, _ = lstate.monte_carlo_estimate(mcs_samples)
                print('pf_ref ', Pf_ref, end=" ")
            else:
                x_mc_norm = np.random.uniform(0, 1, size=(mcs_samples, lstate.input_dim))

            X_uq = torch.tensor(x_mc_norm, dtype=torch.float32)
            
            Y_uq = model.predictive_uq(X_uq)
            y_mean = Y_uq.mean(dim=1)
            pf_sumo = (((y_mean<0).sum()) / torch.tensor(mcs_samples)).item()
            print('pf_surrogate ' , pf_sumo )
            results_dict['pf_'+ str(len(x_train))] = pf_sumo

            x_norm = act_train.get_active_points(x_train, X_uq, Y_uq, active_points)
            x_scaled = isoprob_transform(x_norm, lstate.marginals)
            y_scaled = lstate.eval_lstate(x_scaled)

        results_dict[str(len(x_train)) + '_doepoints'] = x_train, y_train
        # Storing seed for reproducibility
        results_dict['seed'] = seed_experiment
        results_dict['config'] = config_dict

        with open(results_dir + '/' + results_file+ "Res"+ str(exp+1)+ "_" + config_dict['l_state'] + "_" + date_time_stamp + ".pkl", 'wb') as file_id:
            pickle.dump(results_dict, file_id)

        with open(results_dir + '/' + results_file+ "Mod"+ str(exp+1)+ "_" + config_dict['l_state'] + "_" + date_time_stamp + ".pkl", 'wb') as file_id:
            pickle.dump(model, file_id)

        print('End training '+str(exp+1))