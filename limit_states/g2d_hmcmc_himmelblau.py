import numpy as np
from scipy.stats import norm
from scipy.stats import qmc
from doepy import build
import pandas as pd 

'''Example 4: The Himmelblau Function Ref. (HMCMC High Dim, Prof. Kostas, pag. 16) 
which is particularly suitable for reliability examples with multiple separated failure domains. 
x1 and x2 are assumed to be independent standard normal random variables and 
the constant beta is used to define different levels of the failure probability. 
(beta = 95 for Ref. PF=1.65E-4)   (beta = 50 for Ref. PF=2.77E-7)'''

class g2d_himmelblau():
    def __init__(self):
        self.input_dim = 2
        self.output_dim = 1
        self.marginals = {'x1': [0, 1, 'normal'],
                          'x2': [0, 1, 'normal']}
        '''mean(or min), std(or max), marginal_distrib'''

    def eval_lstate(self, x):
        beta = 95 
        term1 = (((0.75*x[:,0] - 0.5)**2 / 1.81) + ((0.75*x[:,1] - 0.5) / 1.81) - 11)**2
        term2 = (((0.75*x[:,0] - 1.0)/ 1.81) + ((0.75*x[:,1] - 0.5)**2 / 1.81) - 7)**2
        g = term1 + term2 - beta
        return g    

    def monte_carlo_estimate(self, n_samples):
        n_mcs = int(n_samples)
        x_mc = np.random.normal(0, 1, size=(n_mcs, self.input_dim))
        y_mc = self.eval_lstate(x_mc)
        Pf_ref = np.sum(y_mc < 0) / n_mcs
        B_ref = - norm.ppf(Pf_ref)
        return Pf_ref, B_ref, x_mc, y_mc

    def get_doe_points(self, n_samples=10, method='lhs'):
        n_passive = int(n_samples)

        if method == 'lhs':
            exp_norm = {}
            for var_name in range(self.input_dim):
                exp_norm['x'+ str(var_name+1)] = [0.001, 0.999]    #design domain for each variable (uniform)
            x_doe = build.space_filling_lhs(exp_norm , num_samples = n_passive)  #Latin hypercube sampling
            x_norm = pd.DataFrame.to_numpy(x_doe)
            x_scaled = self.isoprob_transform(x_norm, self.marginals)
            y_scaled = self.eval_lstate(x_scaled)

        return x_norm, x_scaled, y_scaled

    def isoprob_transform (self, x_normalised, marginals):
        x_scaled = np.zeros((len(x_normalised), self.input_dim))

        for margin in range (0, self.input_dim):
            var = 'x' + str (margin + 1)
            if marginals[var][2] == 'normal':
                loc_ = marginals[var][0]
                scale_ = marginals[var][1]
                x_scaled[:, margin] = norm.ppf(x_normalised[:, margin], loc=loc_, scale=scale_)

            elif marginals[var][2] == 'uniform':
                loc_ = marginals[var][0]
                scale_ = marginals[var][1]
                x_scaled[:, margin] = uniform.ppf(x_normalised[:, margin], loc=loc_, scale=scale_-loc_)

            elif marginals[var][2] == 'lognormal':
                xlog_mean = marginals[var][0]
                xlog_std = marginals[var][1]
                gaussian_param = self.convert_lognormal(xlog_mean, xlog_std)
                x_scaled[:, margin] = lognorm.ppf(x_normalised[:, margin], s=gaussian_param[1], scale=xlog_mean) 
        
        return x_scaled
    
    def convert_lognormal(self, mean_ln, std_ln):
        gaussian_param = np.zeros(2)

        SigmaLogNormal = np.sqrt( np.log(1+(std_ln/mean_ln)**2))
        MeanLogNormal = np.log( mean_ln ) - SigmaLogNormal**2/2

        gaussian_param[0] = MeanLogNormal
        gaussian_param[1] = SigmaLogNormal

        return gaussian_param