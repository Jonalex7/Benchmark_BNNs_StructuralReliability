import torch
import torch.nn as nn

class FFN_block(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FFN_block, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Input layer to hidden layer
        self.relu = nn.ReLU()  # Activation function
        # self.dropout = nn.Dropout(dropout_prob)  # Dropout layer
        self.fc2 = nn.Linear(hidden_size, output_size)  # Hidden layer to output layer

    def forward(self, x):
        # Forward pass
        x = self.fc1(x)
        x = self.relu(x)
        # x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class Ensemble(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_models):
        super(Ensemble, self).__init__()
        self.num_models = num_models
        self.models = nn.ModuleList([FFN_block(input_size, hidden_size, output_size) for _ in range(num_models)])
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def train_ensemble(self, train_loader, num_epochs, verbose=0):
        for epoch in range(num_epochs):
            for inputs, targets in train_loader:
                self.optimizer.zero_grad()
                ensemble_outputs = torch.stack([model(inputs) for model in self.models])
                ensemble_mean = torch.mean(ensemble_outputs, dim=0)
                loss = self.criterion(ensemble_mean, targets)
                loss.backward()
                self.optimizer.step()
            if not verbose == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')
        print("train_loss = %f," % (loss.item()), end=" ")

    def predict_ensemble(self, data_loader):
        self.eval()
        ensemble_predictions = []
        with torch.no_grad():
            for inputs, _ in data_loader:
                ensemble_outputs = torch.stack([model(inputs) for model in self.models])
                ensemble_mean = torch.mean(ensemble_outputs, dim=0)
                ensemble_predictions.append(ensemble_mean)
        self.train()
        return torch.cat(ensemble_predictions, dim=0)
    
    def predictive_uq(self, x):
        x_len = len(x)
        y_random = torch.empty(x_len, self.num_models)
        idx = 0
        with torch.no_grad():
            for model in self.models:
                y_random[:, idx] = model(x).reshape(-1)
                idx += 1
        return y_random