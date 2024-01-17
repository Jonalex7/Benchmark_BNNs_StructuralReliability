import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# Bayesian neural network with class to build Bayesian Layers with Pytorch

class BayesianLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(BayesianLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight_mu, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.weight_rho, -6)  # Initialize log variance to a small value
        nn.init.zeros_(self.bias_mu)
        nn.init.constant_(self.bias_rho, -6)

    def forward(self, x):
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        epsilon_weight = Variable(torch.randn_like(weight_sigma))
        epsilon_bias = Variable(torch.randn_like(bias_sigma))
        weight = self.weight_mu + weight_sigma * epsilon_weight
        bias = self.bias_mu + bias_sigma * epsilon_bias
        output = F.linear(x, weight, bias)
        return output #, weight, bias
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}'
    
class BayesianNeuralNetwork(nn.Module):
    def __init__(self, input_dim, width, depth, output_dim):
        super(BayesianNeuralNetwork, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.width = width
        self.depth = depth

        layers = [BayesianLayer(input_dim, width), nn.ReLU()]
        for i in range(depth - 1):
            layers.append(BayesianLayer(width, width))
            layers.append(nn.ReLU())
        layers.append(BayesianLayer(width, output_dim))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
    
class BNN_BayesBackProp(nn.Module):
    def __init__(self, input_size, hidden_sizes, hidden_layers, output_size):
        super(BNN_BayesBackProp, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.hidden_layers = hidden_layers
        self.output_size = output_size

        self.model = BayesianNeuralNetwork(self.input_size, self.hidden_sizes, self.hidden_layers, self.output_size)

    def train(self, train_loader, num_epochs=10, lr=1e-3, kl_scale=10, verbose=0):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in range(num_epochs):
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                output_mu = self.model(inputs)
                recon_loss = F.mse_loss(output_mu, targets, reduction='mean')
                kl_loss = 0

                for layer in self.model.block.children():
                    if isinstance(layer, BayesianLayer):
                        weight_mu = layer.weight_mu
                        weight_rho = layer.weight_rho
                        bias_mu = layer.bias_mu
                        bias_rho = layer.bias_rho
                        # Compute KL divergence for weights
                        kl_loss += 0.5 * (torch.sum(weight_mu**2) + torch.sum(torch.log1p(torch.exp(weight_rho)) - weight_rho) - weight_mu.numel())
                        # Compute KL divergence for biases
                        kl_loss += 0.5 * (torch.sum(bias_mu**2) + torch.sum(torch.log1p(torch.exp(bias_rho)) - bias_rho) - bias_mu.numel())
                
                kl_loss = kl_loss / kl_scale
                loss = recon_loss + kl_loss
                loss.backward()
                optimizer.step()

            if not verbose == 0: 
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss MSE: {recon_loss}, Loss KL: {kl_loss}')
            
        # print(f'Epoch [{epoch+1}/{num_epochs}], total training loss: {loss.item()}, end=" "')
        print (f'Epoch [{epoch+1}/{num_epochs}], MSE: {recon_loss}, KL:{kl_loss}' , end=" ")

    def predictive_uq (self, x, n_sim):
        x_len = len(x)
        y_random = torch.empty(x_len, n_sim)

        with torch.no_grad():
            for i in range(n_sim):
                y_random[:, i] = self.model(x).reshape(-1)
                
        # mean_uq, std_uq = torch.mean(y_random, 1), torch.std(y_random, 1)
        return y_random
    
    def count_parameters(self, model):   #method to count the number of parameters in the network
        p_count = 0
        for p in model.parameters():
            p_count += len(p)
        return p_count
    

