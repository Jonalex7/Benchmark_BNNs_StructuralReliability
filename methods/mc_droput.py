import torch
import torch.nn as nn

class MLP_dropout(nn.Module):
    def __init__(self, input_dim, width, depth, output_dim, dropout_prob):
        super(MLP_dropout, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.width = width
        self.depth = depth
        self.activation = nn.ReLU()

        layers = [nn.Linear(input_dim, width), self.activation, nn.Dropout(dropout_prob)]
        for i in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout_prob))
        layers.append(nn.Linear(width, output_dim))

        self.block = nn.Sequential(*layers)

        # Initialize the linear layers with Xavier initialization
        layer = 0
        for i in range(depth+1):
            nn.init.xavier_uniform_(self.block[layer].weight)
            nn.init.constant_(self.block[layer].bias, 0)
            layer = layer + 3

    def forward(self, x):
        return self.block(x)
    
class NeuralNetworkWithDropout(nn.Module):
    def __init__(self, input_size, hidden_sizes, hidden_layers, output_size, args):
        super(NeuralNetworkWithDropout, self).__init__()
        self.dropout_prob = args['dropout_probability']
        self.weight_decay = args['weight_decay']
        self.n_sim = args['n_simulations']
        self.case_file = args['case']
        self.model = MLP_dropout(input_size, hidden_sizes, hidden_layers, output_size, self.dropout_prob)

    def train(self, train_loader, test_loader, num_epochs, lr, patience=10 ,verbose=0):
        self.criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=self.weight_decay)
        best_val_loss = torch.inf
        train_avg_loss = []
        test_avg_loss = []

        for epoch in range(num_epochs):
            train_losses = []
            test_losses = []
        
            #Training Loss
            for inputs, labels in train_loader:

                outputs = self.model(inputs)  # Forward pass
                loss = self.criterion(outputs, labels)  # Calculate the loss
                train_losses.append(loss.detach().item())
                
                optimizer.zero_grad()  # Zero the gradients
                loss.backward()  # Backpropagation
                optimizer.step()  # Update the weights

            #Validation Loss
            with torch.no_grad():
                for inputs, labels in test_loader:
                    outputs = self.model(inputs)  # Forward pass
                    loss = self.criterion(outputs, labels)
                    test_losses.append(loss.item())             
            
            train_avg_loss.append(sum(train_losses)/len(train_losses))
            test_avg_loss.append(sum(test_losses)/len(test_losses))

            if verbose is not None:
                print(f'Epoch [{epoch + 1}/{num_epochs}], train_l: {train_avg_loss[-1]:.2E}, test_l:{test_avg_loss[-1]:.2E}')

            # Check for early stopping
            best_model_name = 'best_model_dropout_' + self.case_file +'.pth'

            if test_avg_loss[-1] < best_val_loss:
                best_val_loss = test_avg_loss[-1]
                torch.save(self.model.state_dict(), best_model_name)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch+1}', end=" ")
                    break

        # # print("train_loss = %f" % (loss.item()), end=" ")
        print(f'Epoch [{epoch + 1}/{num_epochs}], train_l: {train_avg_loss[-patience-1]:.2E}, test_l: {test_avg_loss[-patience-1]:.2E}')

        # Load the best model
        self.model.load_state_dict(torch.load(best_model_name))
        return train_avg_loss, test_avg_loss

    def predictive_uq(self, x):
        x_len = len(x)
        y_random = torch.empty(x_len, self.n_sim)
        with torch.no_grad():
            for i in range(self.n_sim):
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
        
    def forward(self, x):
        with torch.no_grad():
            return self.model(x)