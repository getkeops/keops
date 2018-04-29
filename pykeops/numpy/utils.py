import numpy as np


def squared_distances(x, y):
    return np.sum((x[:,np.newaxis,:] - y[np.newaxis,:,:]) ** 2, axis=2)

def differences(x, y):
    return (x.T[:,:,np.newaxis] - y.T[:,np.newaxis,:])

def np_kernel(x, y, s, kernel) :
    sq = squared_distances(x, y)
    if   kernel == "gaussian"  : return np.exp( -sq / (s*s))
    elif kernel == "laplacian" : return np.exp( -np.sqrt(sq) / s)
    elif kernel == "cauchy"    : return  1. / ( 1 + sq / (s*s) )
    elif kernel == "inverse_multiquadric" : return np.sqrt(  1. / ( s*s + sq ) )

def grad_np_kernel(x, y, s, kernel) :
    sq = squared_distances(x, y)
    if   kernel == "gaussian"  : return - np.exp(-sq / (s*s)) / (s*s)
    elif kernel == "laplacian" : t = -np.sqrt(sq / (s*s)) ; return  np.exp(t) / (2*s*s*t)
    elif kernel == "cauchy"    : return -1. / (s * (sq/(s*s) + 1) )**2 
    elif kernel == "inverse_multiquadric"    : return -.5 / (sq + s**2) **1.5 

def chain_rules(q,ax,by,Aa,p):
    res = np.zeros(ax.shape).astype('float32')
    for i in range(ax.shape[1]):
        #Computation of 2*|x_i -x_j|*exp(-|x_i -x_j|^2/(lam^2))/(lam^2)
        ximyj = (np.tile(ax[:,i],[by.shape[0],1]).T - np.tile(by[:,i],[ax.shape[0],1])) 
        res[:,i] = np.sum(q * ((2 * ximyj * Aa) @ p),axis=1)
    return res
