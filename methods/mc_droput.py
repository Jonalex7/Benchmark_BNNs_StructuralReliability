import torch
import torch.nn as nn

class NeuralNetworkWithDropout(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob):
        super(NeuralNetworkWithDropout, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Input layer to hidden layer
        self.relu = nn.ReLU()  # Activation function
        self.dropout = nn.Dropout(dropout_prob)  # Dropout layer
        self.fc2 = nn.Linear(hidden_size, output_size)  # Hidden layer to output layer

    def forward(self, x):
        # Forward pass
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def train(self, train_loader, num_epochs, learning_rate, verbose=0):
        #criterion = nn.CrossEntropyLoss()
        criterion = nn.MSELoss()
        #optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            for inputs, labels in train_loader:
                optimizer.zero_grad()  # Zero the gradients
                outputs = self(inputs)  # Forward pass
                loss = criterion(outputs, labels)  # Calculate the loss
                loss.backward()  # Backpropagation
                optimizer.step()  # Update the weights
            if not verbose == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

    def predictive_uq(self, x, n_sim):
        x_len = len(x)
        y_random = torch.empty(x_len, n_sim)
        with torch.no_grad():
            for i in range(n_sim):
                y_random[:, i] = self.forward(x).reshape(-1)
        return y_random

    def evaluate(self, test_loader):
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        return accuracy