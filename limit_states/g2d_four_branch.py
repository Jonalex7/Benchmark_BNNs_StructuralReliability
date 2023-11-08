def g2D_four_branch(x):
    """
    Schöbi et al. , ASCE J. Risk Unc. (2016)
    The four-branch function is a common benchmark in structural reliability analysis that describes
    the failure of a series system with four distinct component limit states. Its mathematical
    formulation reads (Waarts, 2000; Schueremans and Van Gemert, 2005a,b):
    where the input variables are modeled by two independent Gaussian random variables

    Parameters
    ----------
        x : numpy.array of float(s)
            Values of independent variables: columns are the different parameters/random variables (x1, x2,...xn) and rows are different parameter/random variables sets for different calls.

    Returns
    -------
        g_val_sys : numpy.array of float(s)
            Performance function value for the system.
        g_val_comp : numpy.array of float(s)
            Performance function value for each component.
        msg : str
            Accompanying diagnostic message, e.g. warning.
    """
    import numpy as np
    # expected number of random variables/columns
    nrv_e = 2

    g, g1, g2, g3, g4 = float('nan'), float('nan'), float('nan'), float('nan'), float('nan')
    msg = 'Ok'
    x = np.array(x, dtype='f')

    n_dim = len(x.shape)
    if n_dim == 1:
        x = np.array(x)[np.newaxis]
    elif n_dim > 2:
        msg = 'Only available for 1D and 2D arrays.'
        return float('nan'), float('nan'), msg

    nrv_p = x.shape[1]
    if nrv_p != nrv_e:
        msg = f'The number of random variables (x, columns) is expected to be {nrv_e} but {nrv_p} is provided!'
    else:
        g1 = 3 + 0.1*(x[:, 0] - x[:, 1])**2 - (x[:, 0] + x[:, 1])/np.sqrt(2)
        g2 = 3 + 0.1*(x[:, 0] - x[:, 1])**2 + (x[:, 0] + x[:, 1])/np.sqrt(2)
        g3 = (x[:, 0] - x[:, 1]) + (6/np.sqrt(2))
        g4 = (x[:, 1] - x[:, 0]) + (6/np.sqrt(2))
        g = np.amin(np.stack((g1, g2, g3, g4)), 0)

    g_val_sys = g
    g_val_comp = np.stack((g1, g2, g3, g4))
    return g_val_sys #, g_val_comp, msg