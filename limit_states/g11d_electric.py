import numpy as np
# Reliability analysis and optimal design under uncertainty Focus on adaptive surrogate-based approaches
# Jean-Marc Bourinet
# A–1 Example 1
# This reliability problem is studied by Kouassi et al. (2016) in the field on electromagnetic compatibility.
# It investigates a lossy transmission line of length L, diameter d and attenuation coefficient alpha such
# as defined by Rannou et al. (2002)

def example_electric(x):
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

    #MARGINALS DEFINITION--------------------------------------------------------------------
    # # X1 = L (m) mean=4.2    cov=0.10   lognormal/R>0
    # x1_mean = 4.2
    # x1_std = x1_mean * 0.1
    # normal_std_1 = np.sqrt(np.log(1 + (x1_std/x1_mean)**2))
    # normal_mean_1 = np.log(x1_mean) - normal_std_1**2 / 2

    # # X2 = h (m) mean=0.02    cov=0.10 lognormal /R>0
    # x2_mean = 0.02
    # x2_std = x2_mean * 0.1
    # normal_std_2 = np.sqrt(np.log(1 + (x2_std/x2_mean)**2))
    # normal_mean_2 = np.log(x2_mean) - normal_std_2**2 / 2

    # # X3 = d (m) mean=0.001  cov=0.05 lognormal /R>0
    # x3_mean = 0.001
    # x3_std = x3_mean * 0.05
    # normal_std_3 = np.sqrt(np.log(1 + (x3_std/x3_mean)**2))
    # normal_mean_3 = np.log(x3_mean) - normal_std_3**2 / 2

    # # X4 = ZL () mean=1000    cov=0.20 lognormal /R>0
    # x4_mean = 1000
    # x4_std = x4_mean * 0.2
    # normal_std_4 = np.sqrt(np.log(1 + (x4_std/x4_mean)**2))
    # normal_mean_4 = np.log(x4_mean) - normal_std_4**2 / 2

    # # X5 = Z0 () mean=50   cov=0.05 lognormal /R>0
    # x5_mean = 50
    # x5_std = x5_mean * 0.05
    # normal_std_5 = np.sqrt(np.log(1 + (x5_std/x5_mean)**2))
    # normal_mean_5 = np.log(x5_mean) - normal_std_5**2 / 2

    # # X6 = ae (V/m) mean=1  cov=0.20 lognormal /R>0
    # x6_mean = 1
    # x6_std = x6_mean * 0.2
    # normal_std_6 = np.sqrt(np.log(1 + (x6_std/x6_mean)**2))
    # normal_mean_6 = np.log(x6_mean) - normal_std_6**2 / 2

    # # X7 = theta_e (rad) mean=pi/4    cov=0.577 uniform / [0,pi/2]
    # x7_min = 0
    # x7_max = np.pi / 2

    # # X8 = theta_p (rad) mean=pi/4    cov=0.577 uniform / [0,pi/2]
    # x8_min = 0
    # x8_max = np.pi / 2

    # # X9 = phi_p (rad) mean=pi   cov=0.577 uniform / [0,pi*2]
    # x9_min = 0
    # x9_max = np.pi*2

    # # X10 = f (MHz) mean=30    cov=0.096 uniform / [25 ,35]
    # x10_min = 25.
    # x10_max = 35.

    # # X11 = alpha (-) mean=0.0010  cov=0.289 uniform / [0.0005 , 0.0015]
    # x11_min = 0.0005
    # x11_max =  0.0015

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