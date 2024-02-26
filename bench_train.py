import os
import collections
from copy import deepcopy
import torch
import torch.nn as nn
from datetime import datetime
import numpy as np
from sklearn.base import BaseEstimator
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import sys
import pickle
import matplotlib.pyplot as plt
from utils.data import normalize_data, denormalize_data
import scipy
import yaml

from methods import REGISTRY as met_REGISTRY
from utils.data import get_dataloader
from utils.hyperoptim import PyTorchEstimator_NN

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
    
def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

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
                # data_set = scipy.io.loadmat(f, simplify_cells=True)
                data_set = pickle.load(f)
            except Exception as e:
                print("Error loading .pkl file:", e)
        return data_set, file_name
    
def dict_to_text(info_dict):
    text = ""
    for key, value in info_dict.items():
        text += f"{key}: {value}\n"
    return text
    
if __name__ == "__main__":
    params = deepcopy(sys.argv)
    # Get the defaults from default.yaml
    with open(
        os.path.join(os.path.dirname(__file__), "config", "default_bench.yaml"), "r"
    ) as f:
        try:
            config_dict = yaml.full_load(f)
        except yaml.YAMLError as exc:
            assert False, "default_bench.yaml error: {}".format(exc)

    # Load algorithm and env base configs
    model, method_name = _get_config(params, "--method", "mod")
    config_dict = recursive_dict_update(config_dict, model)
    
    data, file_name = _get_data(params, "--data", "data")

# Directory to save results
results_dir = 'bench_cases/results/'+'/'+ file_name +'/'+ method_name

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

#loading default_bench arguments -------------------------------------------------

batch_size = config_dict['batch_size']
args = config_dict['args']
patience = config_dict['patience']
training_epochs = config_dict['training_epochs']
verbose = config_dict['verbose']
split_train_test = config_dict['split_train_test']

method = met_REGISTRY[config_dict['method']]

training_sets = len(data['training'])
n_replications = config_dict['n_replications']

date_time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

if config_dict['validation'] is not None:
    valid_samples = config_dict['validation_size']   #size of the validation data set
    #validation samples
    x_valid = data['validation'][0] # feature matrix, shape (100_000,3)
    y_valid = data['validation'][1] # dependent variable, shape (100_000,)
    # x_valid = torch.tensor(x_valid[:valid_samples], dtype=torch.float32)
    # y_valid = torch.tensor(y_valid[:valid_samples], dtype=torch.float32).reshape(-1, 1)

for set in range(training_sets):

    for rep in range(n_replications):

        results_file = {}

        # Seed definition
        if config_dict['seed'] is not None:
            seed_experiment = config_dict['seed']
        else:
            seed_experiment = np.random.randint(0, 2**30 - 1)
        np.random.seed(seed_experiment)
        torch.manual_seed(seed_experiment)

        #loading data set-----------------------------------------------------------------
        #training samples 

        set=6

        x_train = data['training'][set][rep][0]
        y_train = data['training'][set][rep][1]

        print('Data set: ', file_name, '-----------------------------------------------------')
        print('Training set size: ', x_train.shape[0], end=" ")
        print('Replication: ', rep+1)

        config_dict['train_set_size'] = (set, x_train.shape[0])
        config_dict['replication'] = rep
        args['case'] = config_dict['data_case'] = file_name

        # x_train = torch.tensor(x_train, dtype=torch.float32)
        # y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)

        input_dim = x_train.shape[1]
        output_dim = 1

        x_norm, x_mean, x_var = normalize_data(x_train)
        y_norm, y_mean, y_var = normalize_data(y_train)

        # p_norm = 2
        # x_denom = x_train.norm(p_norm, dim=0, keepdim=True)
        # y_denom = y_train.norm(p_norm, dim=0, keepdim=True)

        # x_norm = x_train / x_denom
        # y_norm = y_train / y_denom

        # args['kl_scale'] = x_train.shape[0] * split_train_test

        if config_dict['optimize_network'] is not None:
            print('Optimizing hyperparameters')

            args['split_train_test'] = split_train_test
            args['batch_size'] = batch_size
            args['training_epochs'] = training_epochs
            args['patience'] = patience
            args['verbose'] = verbose

            # # Define the search space for hyperparameters
            param_space = {
                'lr': Real(1e-5, 1e-1, prior='log-uniform'),
                'hidden_size': Integer(100, 500, prior='uniform'),
                'hidden_layers': Integer(3, 5, prior='uniform'),
                # 'kl_scale': Integer(100, 1000, prior='uniform')}
                'drop_prob': Real(0.0005, 0.2, prior='uniform') }

            # Perform Bayesian optimization
            bayes_cv_tuner = BayesSearchCV(
                PyTorchEstimator_NN(method=method, args=args),
                search_spaces=param_space,
                n_iter=20,
                cv=3,
                random_state=42)
              
            #Optimizing the hyperparameters ------------------------------
            # np.int = int
            bayes_cv_tuner.fit(x_norm, y_norm)

            # Get the best hyperparameters
            best_params = bayes_cv_tuner.best_params_
            print("Best hyperparameters:", best_params)

            hidden_layers = config_dict['opt_hidden_layers'] = best_params['hidden_layers']
            hidden_sizes = config_dict['opt_hidden_sizes'] = best_params['hidden_size']
            lr = config_dict['opt_learning_rate'] = best_params['lr']

            if method == met_REGISTRY['bnnbpp']:
                args['kl_scale'] = config_dict['opt_kl_scale'] = best_params['kl_scale']
            elif method == met_REGISTRY['dropout']:
                args['dropout_probability'] = config_dict['opt_drop_prob'] = best_params['drop_prob']

        else:
            #hyperparameters from config. file------------------------------
            hidden_layers = config_dict['hidden_layers']
            hidden_sizes = config_dict['hidden_sizes']
            lr = config_dict['learning_rate']

        #Creating and training the model
        model = method(input_size=input_dim, hidden_sizes=hidden_sizes, 
                    hidden_layers=hidden_layers, output_size=output_dim, 
                    args=args)

        train_loader, test_loader = get_dataloader(x_norm, y_norm, split_train_test, batch_size)
        
        train_hist, valid_hist = model.train(train_loader, test_loader, num_epochs=training_epochs, 
                                            lr=lr, patience=patience, verbose=verbose)

        results_file['seed'] = seed_experiment
        results_file['config'] = config_dict
        results_file['train_hist'] = train_hist
        results_file['valid_hist'] = valid_hist
        results_file['model'] = model
        results_file['x_mean_var'] = x_mean, x_var
        results_file['y_mean_var'] = y_mean, y_var

        if config_dict['validation'] is not None:
            # Validation error - RSUQ metric, Relative MSE
            print('Estimating Rel. MSE...')
            Y_uq_val = model.predictive_uq(normalize_data(x_valid, mean=x_mean, variance=x_var))
            y_mean_norm = Y_uq_val.mean(dim=1).reshape(-1,1)
            y_mean_val= denormalize_data(y_mean_norm, mean=y_mean, variance=y_var)

            rel_mse = model.criterion(y_valid, y_mean_val)/(torch.var(y_valid))
            results_file['rel_mse'] = rel_mse.item()

            print(f'Relative MSE on validation data set with {valid_samples} samples: {rel_mse.item():.2E}')

        if config_dict['plot'] is not None:
            cm = 1/2.54  # centimeters in inches
            plt.rcParams["font.family"] = "Times New Roman"
            plt.rcParams["font.size"] = 12

            info_text = dict_to_text(config_dict)
            #Saving plot for losses history
            fig1, axs = plt.subplots(1, 1 , figsize=(16*cm, 11*cm), dpi=100, facecolor='w', edgecolor='k')
            plt.subplots_adjust(left=0.12, right=.95, top=0.90, bottom=0.15, hspace = 0.65, wspace=0.6)

            fig1.suptitle(method_name+', ' + ' training samples: '+ str(x_train.shape[0])+ ', replication: '+ str(rep+1) )
            axs.plot(train_hist, label='train loss')
            axs.plot(valid_hist, label='test loss')
            axs.text(0.01, 1.0, info_text, transform=axs.transAxes, fontsize=10, verticalalignment='top')
            axs.set_xlabel('epochs')
            axs.set_yscale('log')
            plt.grid(linestyle='--', linewidth=0.5,  which='Both')
            plt.legend()
            # plt.show()
            fig1.savefig(results_dir+'/'+'set_'+str(set+1)+'_rep_'+str(rep+1)+"_"+date_time_stamp+'.jpg')

        #Saving results file
        with open(results_dir +'/'+'set_'+str(set+1)+'_rep_'+str(rep+1) + "_" + date_time_stamp + ".pkl", 'wb') as file_id:
            pickle.dump(results_file, file_id)