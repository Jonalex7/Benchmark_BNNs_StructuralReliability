import torch
import torch.nn as nn

class MLP_dropout(nn.Module):
    def __init__(self, input_dim, width, depth, output_dim, dropout_prob):
        super(MLP_dropout, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.width = width
        self.depth = depth

        layers = [nn.Linear(input_dim, width), nn.ReLU(), nn.Dropout(dropout_prob)]
        for i in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))
        layers.append(nn.Linear(width, output_dim))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
    
class NeuralNetworkWithDropout(nn.Module):
    def __init__(self, input_size, width, depth, output_size, dropout_prob):
        super(NeuralNetworkWithDropout, self).__init__()
        self.model = MLP_dropout(input_size, width, depth, output_size, dropout_prob)

    def train(self, train_loader, num_epochs, learning_rate, verbose=0):
        #criterion = nn.CrossEntropyLoss()
        criterion = nn.MSELoss()
        #optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            for inputs, labels in train_loader:
                optimizer.zero_grad()  # Zero the gradients
                outputs = self.model(inputs)  # Forward pass
                loss = criterion(outputs, labels)  # Calculate the loss
                loss.backward()  # Backpropagation
                optimizer.step()  # Update the weights
            if not verbose == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')
        print("train_loss = %f" % (loss.item()), end=" ")

    def predictive_uq(self, x, n_sim):
        x_len = len(x)
        y_random = torch.empty(x_len, n_sim)
        with torch.no_grad():
            for i in range(n_sim):
                y_random[:, i] = self.model(x).reshape(-1)
        return y_random

    def evaluate(self, test_loader):
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        return accuracy