import torch
from torch.utils.data import DataLoader, TensorDataset

def get_dataloader(X, Y, input_dim, output_dim, train_test_split, batch_size):
    # Torch array?
    if not type(X) == torch.Tensor:
        inputs = torch.tensor(X, dtype=torch.float32)
    else:
        inputs = X
    
    if not type(Y) == torch.Tensor:
        targets = torch.tensor(Y.reshape(-1, output_dim), dtype=torch.float32)
    else:
        targets = Y.reshape(-1, output_dim)

    # Split the data into training and testing sets
    train_size = int(train_test_split * len(inputs))
    #test_size = len(inputs) - train_size

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