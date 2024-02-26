import torch
from sklearn.base import BaseEstimator
from utils.data import get_dataloader
from methods import REGISTRY as met_REGISTRY

# Define the function to optimize
class PyTorchEstimator_NN(BaseEstimator):
    def __init__(self, lr=0.01, hidden_size=10, hidden_layers=1, method=None, args=None):
        self.lr = lr
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.method = method
        self.args = args

        if self.method == met_REGISTRY['bnnbpp']:
            self.kl_scale = self.args['kl_scale']
        elif self.method == met_REGISTRY['dropout']:
            self.drop_prob = self.args['dropout_probability']
        
    def fit(self, X, y):
        # print(method)
        input_size = X.shape[1] if len(X.shape) > 1 else 1
        output_size = y.shape[1] if len(y.shape) > 1 else 1

        if self.method == met_REGISTRY['bnnbpp']:
            self.args['kl_scale'] = self.kl_scale
            # print('searching param: ', self.hidden_size, self.hidden_layers, self.lr, self.kl_scale)
        elif self.method == met_REGISTRY['dropout']:
            self.args['dropout_probability'] = self.drop_prob
            # print('searching param: ', self.hidden_size, self.hidden_layers, self.lr, self.drop_prob)
        
        # Initialize the model
        self.model = self.method(input_size=input_size, hidden_sizes=self.hidden_size, 
                hidden_layers=self.hidden_layers, output_size=output_size, 
                args=self.args)

        train_loader_, self.test_loader_ = get_dataloader(X, y, self.args['split_train_test'], self.args['batch_size'])

        # Train the model
        _, _ = self.model.train(train_loader_, self.test_loader_, num_epochs=self.args['training_epochs'], 
                                lr=self.lr, patience=self.args['patience'], verbose=self.args['verbose'])

    def predict(self, X):
        with torch.no_grad():
            return self.model(X)

    # def score(self, X, y):
    #     with torch.no_grad():
    #         predictions = self.model(X)
    #         mse = self.model.criterion(predictions, y)
    #     return -mse  # to Minimize MSE
    
    def score(self, X, y):
        test_avg_loss = []
        with torch.no_grad():
            test_losses=[]
            for inputs, labels in self.test_loader_:
                outputs = self.model(inputs)  # Forward pass
                loss = self.model.criterion(outputs, labels)
                test_losses.append(loss)
            test_avg_loss.append(sum(test_losses)/len(test_losses))
        return - test_avg_loss[-1] # to Minimize average MSE

    def get_params(self, deep=True):
        return {
            'args' : self.args,
            'method': self.method
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    