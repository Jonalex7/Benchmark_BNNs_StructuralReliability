import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import copy

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
        # nn.init.kaiming_normal_(self.weight_mu, mode='fan_in', nonlinearity='relu')
        nn.init.xavier_uniform_(self.weight_mu)
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

    def kl_divergence(self, mu_q, log_sigma_q, mu_p, log_sigma_p):
        """
        Compute KL divergence between two Gaussian distributions with parameters (mu, log_sigma).
        KL(q||p) = log(sigma_p / sigma_q) + (sigma_q^2 + (mu_q - mu_p)^2) / (2 * sigma_p^2) - 0.5
        """
        kl = log_sigma_p - log_sigma_q + (torch.exp(log_sigma_q)**2 + (mu_q - mu_p)**2) / (2 * torch.exp(log_sigma_p)**2) - 0.5
        return kl.sum()

    def layer_kl_divergence(self):
        weight_prior_mu = torch.zeros_like(self.weight_mu)
        weight_prior_log_sigma = torch.zeros_like(self.weight_rho)
        bias_prior_mu = torch.zeros_like(self.bias_mu)
        bias_prior_log_sigma = torch.zeros_like(self.bias_rho)
        
        kl_weight = self.kl_divergence(self.weight_mu, torch.log1p(torch.exp(self.weight_rho)),
                                  weight_prior_mu, torch.log1p(torch.exp(weight_prior_log_sigma)))
        
        kl_bias = self.kl_divergence(self.bias_mu, torch.log1p(torch.exp(self.bias_rho)),
                                bias_prior_mu, torch.log1p(torch.exp(bias_prior_log_sigma)))
        
        return kl_weight + kl_bias
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}'
    
class BayesianNeuralNetwork(nn.Module):
    def __init__(self, input_dim, width, depth, output_dim):
        super(BayesianNeuralNetwork, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.width = width
        self.depth = depth
        self.activation = nn.ReLU()

        layers = [BayesianLayer(input_dim, width), self.activation]
        for i in range(depth - 1):
            layers.append(BayesianLayer(width, width))
            layers.append(self.activation)
        layers.append(BayesianLayer(width, output_dim))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
    
class BNN_BayesBackProp(nn.Module):
    def __init__(self, input_size, hidden_sizes, hidden_layers, output_size, args):
        super(BNN_BayesBackProp, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.kl_scale = args['kl_scale']
        self.n_sim = args['n_simulations']
        self.weight_decay = args['weight_decay']

        self.model = BayesianNeuralNetwork(self.input_size, self.hidden_sizes, self.hidden_layers, self.output_size)

    def train(self, train_loader, test_loader, num_epochs=10, lr=1e-3, patience=10, verbose=0):
        self.criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=self.weight_decay)
        self.num_epochs = num_epochs
        best_total_test_loss = torch.inf
        train_avg_recon_loss = []
        train_avg_kl_loss = []
        total_train_ave_loss = []
        test_avg_recon_loss = []
        test_avg_kl_loss = []
        total_test_ave_loss = []

        for epoch in range(num_epochs):
            train_recon_losses = []
            train_kl_losses = []
            total_train_loss = []
            test_recon_losses = []           
            test_kl_losses = [] 
            total_test_loss = []

            for inputs, targets in train_loader:
                #reconstruction loss
                output_mu = self.model(inputs)
                recon_loss = self.criterion(output_mu, targets)
                train_recon_losses.append(recon_loss.detach().item())
                #kl divergence loss
                kl_loss = self.get_model_kl_loss()
                kl_loss = kl_loss / self.kl_scale
                train_kl_losses.append(kl_loss.detach().item())

                loss = recon_loss + kl_loss
                total_train_loss.append(loss.detach().item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_avg_recon_loss.append(sum(train_recon_losses)/len(train_recon_losses))
            train_avg_kl_loss.append(sum(train_kl_losses)/len(train_kl_losses))
            total_train_ave_loss.append(sum(total_train_loss)/len(total_train_loss))

            with torch.no_grad():
                for inputs, targets in test_loader:
                    #reconstruction loss
                    outputs = self.model(inputs)
                    recon_loss = self.criterion(outputs, targets)
                    test_recon_losses.append(recon_loss.detach().item())
                    #kl divergence loss
                    kl_loss = self.get_model_kl_loss()
                    kl_loss = kl_loss / self.kl_scale
                    test_kl_losses.append(kl_loss.detach().item())

                    valid_loss = recon_loss + kl_loss
                    total_test_loss.append(valid_loss.detach().item())

                test_avg_recon_loss.append(sum(test_recon_losses)/len(test_recon_losses))
                test_avg_kl_loss.append(sum(test_kl_losses)/len(test_kl_losses))
                total_test_ave_loss.append(sum(total_test_loss)/len(total_test_loss))

            # Check for early stopping
            if total_test_ave_loss[-1] < best_total_test_loss:
                #saving best losses
                best_total_test_loss = total_test_ave_loss[-1]
                best_test_recons = test_avg_recon_loss[-1]
                best_test_kl = test_avg_kl_loss[-1]

                best_total_train_loss = total_train_ave_loss[-1]   
                best_train_recon = train_avg_recon_loss[-1]
                best_train_kl = train_avg_kl_loss[-1]

                #saving best model
                self.best_model = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch [{epoch+1}/{self.num_epochs}]')
                    break

            if verbose is not None:
                print (f'Epoch [{epoch+1}/{self.num_epochs}] Train, MSE: {train_avg_recon_loss[-1]:.2E} + KL:{train_avg_kl_loss[-1]:.2E} = {total_train_ave_loss[-1]:.2E} /' 
                       f' Test, MSE: {test_avg_recon_loss[-1]:.2E} + KL:{test_avg_kl_loss[-1]:.2E} = {total_test_ave_loss[-1]:.2E}')
        
        # Loading the best model
        self.model.load_state_dict(self.best_model)

        #end of training
        print (f'End Train, MSE: {best_train_recon:.2E} + KL:{best_train_kl:.2E} = {best_total_train_loss:.2E} /' 
               f' Test, MSE: {best_test_recons:.2E} + KL:{best_test_kl:.2E} = {best_total_test_loss:.2E}')
        
        return best_total_train_loss, best_total_test_loss

    def predictive_uq (self, x):
        x_len = len(x)
        y_random = torch.empty(x_len, self.n_sim)

        with torch.no_grad():
            for i in range(self.n_sim):
                y_random[:, i] = self.model(x).reshape(-1)
                
        # mean_uq, std_uq = torch.mean(y_random, 1), torch.std(y_random, 1)
        return y_random
    
    def count_parameters(self, model):   #method to count the number of parameters in the network
        p_count = 0
        for p in model.parameters():
            p_count += len(p)
        return p_count

    def get_model_kl_loss(self):
        kl_loss = 0
        for layer in self.model.block.children():
            if isinstance(layer, BayesianLayer):
                kl_loss+=layer.layer_kl_divergence()
        return kl_loss

    def forward(self, x):
        with torch.no_grad():
            return self.model(x)
    

