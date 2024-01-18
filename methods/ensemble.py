import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, width, depth, output_dim):
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.width = width
        self.depth = depth

        layers = [nn.Linear(input_dim, width), nn.ReLU()]
        for i in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(width, output_dim))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
    
class Ensemble(nn.Module):
    def __init__(self, input_size, hidden_size, depth, output_size, num_models):
        super(Ensemble, self).__init__()
        self.num_models = num_models
        self.models = nn.ModuleList([MLP(input_size, hidden_size, depth, output_size) for _ in range(num_models)])

    def train_ensemble(self, train_loader, num_epochs, learning_rate, verbose=0):
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            for inputs, targets in train_loader:
                self.optimizer.zero_grad()
                ensemble_outputs = torch.stack([model(inputs) for model in self.models])
                ensemble_mean = torch.mean(ensemble_outputs, dim=0)  #check loss
                loss = self.criterion(ensemble_mean, targets)
                loss.backward()
                self.optimizer.step()
            if not verbose == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')
        print("train_loss = %f" % (loss.item()), end=" ")

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