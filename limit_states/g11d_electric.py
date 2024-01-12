import numpy as np
from doepy import build
import pandas as pd
from scipy.stats import norm, uniform, lognorm
from utils.data import isoprob_transform

'''Reliability analysis and optimal design under uncertainty Focus on adaptive surrogate-based approaches. Jean-Marc Bourinet A–1 Example 1
This reliability problem is studied by Kouassi et al. (2016) in the field on electromagnetic compatibility.
It investigates a lossy transmission line of length L, diameter d and attenuation coefficient alpha such
as defined by Rannou et al. (2002)'''

class g11d_electric():
    def __init__(self):
        self.input_dim = 11
        self.output_dim = 1

        self.marginals = {'x1': [4.20, 4.20* 0.1, 'lognormal'],
                            'x2': [0.02, 0.02* 0.1, 'lognormal'],
                            'x3': [0.001, 0.001* 0.05, 'lognormal'],
                            'x4': [1000, 1000* 0.2, 'lognormal'],
                            'x5': [50, 50* 0.05, 'lognormal'],
                            'x6': [1, 1* 0.2, 'lognormal'],
                            'x7': [0, np.pi / 2, 'uniform'],
                            'x8': [0, np.pi / 2, 'uniform'],
                            'x9': [0, np.pi * 2, 'uniform'],
                            'x10': [25e6, 35e6, 'uniform'],
                            'x11': [0.0005, 0.0015, 'uniform']}
        '''mean(or min), std(or max), marginal_distrib'''
        
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
    
    def eval_lstate(self, x):
        X = np.array(x, dtype='f')

        L = X[:,0] #L (m)
        h = X[:,1] #h (m)
        d = X[:,2] #d (m)
        ZL = X[:,3] #ZL ()
        Z0 = X[:,4] #Z0 ()
        ae = X[:,5] #ae (V/m)
        theta_e = X[:,6]  #theta_e (rad)
        theta_p = X[:,7]  #theta_p (rad)
        phi_p =  X[:,8]  #phi_p (rad)
        f = X[:,9] #f (MHz)
        alpha = X[:,10] #alpha (-)

        ZC = 60*np.arccosh(2*h/d)
        Beta = 2*np.pi*f / (3e8)
        Gamma = alpha + 1j*Beta

        I1 = (Z0*ZC + ZL*ZC)*np.cosh(Gamma*L) + (ZC**2 + Z0*ZL)*np.sinh(Gamma*L)
        I2 = np.sin(Beta*h*np.cos(theta_p)) / (Beta*h*np.cos(theta_p))
        I3 = 1j*Beta*np.cos(theta_p) * ( -np.sin(theta_e) * np.cos(theta_p) * np.sin(phi_p) + np.cos(theta_e)*np.cos(phi_p) )
        I4 = 0.5* (ZC + Z0) * (np.exp( (Gamma + 1j*Beta*np.sin(theta_p)*np.sin(phi_p))*L ) - 1) / (Gamma + 1j*Beta*np.sin(theta_p)*np.sin(phi_p))
        I5 = 0.5* (ZC - Z0) * (np.exp( -(Gamma - 1j*Beta*np.sin(theta_p)*np.sin(phi_p))*L ) - 1) / (Gamma - 1j*Beta*np.sin(theta_p)*np.sin(phi_p))
        I6 = np.sin(theta_e) * np.sin(theta_p) * (ZC - ( ZC*np.cosh(Gamma*L) + Z0*np.sinh(Gamma*L)) *np.exp(1j*Beta*L*np.sin(theta_p)*np.sin(phi_p)) ) 

        I_t = (2*h*ae / I1) * I2* (I3*(I4-I5) + I6)
        I_x = np.absolute(I_t)
        I_cr = 1.5e-4
        g_i = I_cr - I_x

        return g_i

    '''
    Snippet for testing purposes

    # Function to convert from lognormal to corresponding Gaussian parameters:

    def convert_lognormal(mean_ln, std_ln):
        gaussian_param = np.zeros(2)
        
        SigmaLogNormal = np.sqrt( np.log(1+(std_ln/mean_ln)**2))
        MeanLogNormal = np.log( mean_ln ) - SigmaLogNormal**2/2
        
        gaussian_param[0] = MeanLogNormal
        gaussian_param[1] = SigmaLogNormal
        
        return gaussian_param

    # Definition of random variables:

    samples_ = 1000000

    x = np.zeros((samples_, 11))

    mean_, std_ = convert_lognormal(4.2, 4.2*0.1) #L
    x[:, 0] = np.random.lognormal(mean=mean_, sigma=std_, size=samples_)

    mean_, std_ = convert_lognormal(0.02, 0.02*0.1) #h
    x[:, 1] = np.random.lognormal(mean=mean_, sigma=std_, size=samples_)

    mean_, std_ = convert_lognormal(0.001, 0.001*0.05) #d
    x[:, 2] = np.random.lognormal(mean=mean_, sigma=std_, size=samples_)

    mean_, std_ = convert_lognormal(1000, 1000*0.2) #Z_l
    x[:, 3] = np.random.lognormal(mean=mean_, sigma=std_, size=samples_)

    mean_, std_ = convert_lognormal(50, 50*0.05) #Z_0
    x[:, 4] = np.random.lognormal(mean=mean_, sigma=std_, size=samples_)

    mean_, std_ = convert_lognormal(1, 1*0.2) #E_0
    x[:, 5] = np.random.lognormal(mean=mean_, sigma=std_, size=samples_)

    x[:, 6] = np.random.uniform(0, np.pi/2, samples_) #theta_e (rad)
    x[:, 7] = np.random.uniform(0, np.pi/2, samples_) #theta_p (rad)

    x[:, 8] = np.random.uniform(0, 2*np.pi, samples_) #phi_p (rad) 

    x[:, 9] = np.random.uniform(25e6, 35e6, samples_) #f (MHz)

    x[:, 10] = np.random.uniform(0.0005, 0.0015, samples_) #alpha (-)

    # Checking statistics #
    print(x.mean(axis=0), x.std(axis=0), x.std(axis=0)/x.mean(axis=0)*100)

    output = example_electric(x)

    pf = np.sum(output < 0) / samples_
    print(pf)

    '''