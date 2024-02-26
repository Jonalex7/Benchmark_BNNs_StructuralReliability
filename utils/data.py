import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import norm, uniform, lognorm

def get_dataloader(X, Y, train_test_split, batch_size):
    if not type(X) == torch.Tensor:
        inputs = torch.tensor(X, dtype=torch.float32)
    else:
        inputs = X
    
    if not type(Y) == torch.Tensor:
        targets = torch.tensor(Y.reshape(-1, 1), dtype=torch.float32)
    else:
        targets = Y.reshape(-1, 1)

    # Split the data into training and testing sets
    train_size = int(train_test_split * len(inputs))

    train_inputs = inputs[:train_size]
    train_targets = targets[:train_size]

    test_inputs = inputs[train_size:]
    test_targets = targets[train_size:]

    # Create data loaders for training and testing
    train_dataset = TensorDataset(train_inputs, train_targets)
    test_dataset = TensorDataset(test_inputs, test_targets)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def isoprob_transform (x_normalised, marginals):
    input_dim = x_normalised.shape[1]
    x_normalised = torch.tensor(x_normalised)
    x_scaled = torch.zeros(x_normalised.shape[0], input_dim)

    for margin in range (0, input_dim):
        var = 'x' + str (margin + 1)
        if marginals[var][2] == 'normal':
            loc_ = marginals[var][0]
            scale_ = marginals[var][1]
            x_scaled[:, margin] = torch.tensor(norm.ppf(x_normalised[:, margin], loc=loc_, scale=scale_))

        elif marginals[var][2] == 'uniform':
            loc_ = marginals[var][0]
            scale_ = marginals[var][1]
            x_scaled[:, margin] = torch.tensor(uniform.ppf(x_normalised[:, margin], loc=loc_, scale=scale_-loc_))

        elif marginals[var][2] == 'lognormal':
            xlog_mean = torch.tensor(marginals[var][0])
            xlog_std = torch.tensor(marginals[var][1])
            # converting lognormal mean and std. dev.
            SigmaLogNormal = torch.sqrt( torch.log(1+(xlog_std/xlog_mean)**2))
            MeanLogNormal = torch.log(xlog_mean) - SigmaLogNormal**2/2
            x_scaled[:, margin] = torch.tensor(lognorm.ppf(x_normalised[:, margin], s=SigmaLogNormal, scale=xlog_mean)) 
    
    return x_scaled

def normalize_data(dataset, mean=None, variance=None):

    if mean is not None and variance is not None:
        # Normalize the dataset
        normalized_dataset = (dataset - mean) / torch.sqrt(variance + 1e-8)  # Adding a small epsilon to avoid division by zero
        return normalized_dataset
    
    else:
    # Compute the mean and variance along each dimension
        mean = torch.mean(dataset, dim=0)
        variance = torch.var(dataset, dim=0)
        # Normalize the dataset
        normalized_dataset = (dataset - mean) / torch.sqrt(variance + 1e-8)  # Adding a small epsilon to avoid division by zero
        return normalized_dataset, mean, variance

def denormalize_data(normalized_dataset, mean, variance):
    # Recover the original dataset
    dataset = normalized_dataset * torch.sqrt(variance + 1e-8) + mean  # Adding a small epsilon to avoid division by zero
    
    return dataset