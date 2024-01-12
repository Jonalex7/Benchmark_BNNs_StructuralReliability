import numpy as np
from scipy.stats import norm
from scipy.stats import qmc
from doepy import build
import pandas as pd 

from utils.data import isoprob_transform

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
        x = np.array(x, dtype='f')
        beta = 95 
        term1 = (((0.75*x[:,0] - 0.5)**2 / 1.81) + ((0.75*x[:,1] - 0.5) / 1.81) - 11)**2
        term2 = (((0.75*x[:,0] - 1.0)/ 1.81) + ((0.75*x[:,1] - 0.5)**2 / 1.81) - 7)**2
        g = term1 + term2 - beta
        return g    

    def monte_carlo_estimate(self, n_samples):
        n_mcs = int(n_samples)
        x_mc_norm = np.random.uniform(0, 1, size=(int(n_mcs), self.input_dim))
        x_mc_scaled = isoprob_transform(x_mc_norm, self.marginals)
        y_mc = self.eval_lstate(x_mc_scaled)
        Pf_ref = np.sum(y_mc < 0) / n_mcs
        B_ref = - norm.ppf(Pf_ref)
        return Pf_ref, B_ref, x_mc_norm, y_mc

    def get_doe_points(self, n_samples=10, method='lhs'):
        n_passive = int(n_samples)

        if method == 'lhs':
            exp_norm = {}
            for var_name in range(self.input_dim):
                exp_norm['x'+ str(var_name+1)] = [0.001, 0.999]    #design domain for each variable (uniform)
            x_doe = build.space_filling_lhs(exp_norm , num_samples = n_passive)  #Latin hypercube sampling
            x_norm = pd.DataFrame.to_numpy(x_doe)
            x_scaled = isoprob_transform(x_norm, self.marginals)
            y_scaled = self.eval_lstate(x_scaled)

        return x_norm, x_scaled, y_scaled