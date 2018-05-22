import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + (os.path.sep + '..')*3)

# Standard imports
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import torch
from torch.autograd import grad
from pykeops.torch  import Kernel, kernel_product

plt.ion()

# Choose the storage place for our data : CPU (host) or GPU (device) memory.
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


# Define our dataset =====================================================================
# Three points in the plane R^2
y = torch.tensor( [
    [ .2, .7],
    [ .5, .3],
    [ .7, .5]
    ]).type(dtype)
# Three scalar weights
b = torch.tensor([
    1., 1., .5
    ]).type(dtype)
# Remember : b is not a vector, but a "list of unidimensional vectors"!
b = b.view(-1,1) 

# DISPLAY ================================================================================
# Create a uniform grid on the unit square:
res = 100
ticks  = np.linspace( 0, 1, res+1)[:-1] + .5 / res 
X,Y    = np.meshgrid( ticks, ticks )

# Beware! By default, numpy uses float64 precision whereas pytorch uses float32.
# If you don't convert explicitely your data to compatible dtypes,
# PyTorch or Keops will throw an error.
x = torch.from_numpy(np.vstack( (X.ravel(), Y.ravel()) ).T).contiguous().type(dtype)

def showcase_params( params , title, ind) :
    """Samples "x -> ∑_j b_j * k_j(x - y_j)" on the grid, and displays it as a heatmap."""
    heatmap   = kernel_product(params, x, y, b)
    heatmap   = heatmap.view(res,res).cpu().numpy() # reshape as a "background" image
    
    plt.subplot(2,3,ind)
    plt.imshow(  -heatmap, interpolation='bilinear', origin='lower', 
                vmin = -1, vmax = 1, cmap=cm.RdBu, 
                extent=(0,1,0,1)) 
    plt.title(title, fontsize=20)

plt.figure()

# TEST ===================================================================================
# Let's use a "gaussian" kernel, i.e.
#        k(x_i,y_j) = exp( - WeightedSquareNorm(gamma, x_i-y_j ) )
params = {
    "id"      : Kernel("gaussian(x,y)"),
}

# The type of kernel is inferred from the shape of the parameter "gamma",
# used as a "metric multiplier".
# Denoting D == x.shape[1] == y.shape[1] the size of the feature space, rules are : 
#   - if "gamma" is a vector    (gamma.shape = [K]),   it is seen as a fixed parameter
#   - if "gamma" is a 2d-tensor (gamma.shape = [M,K]), it is seen as a "j"-variable "gamma_j"
#
#   - if K == 1 , gamma is a scalar factor in front of a simple euclidean squared norm :
#                 WeightedSquareNorm( g, x-y ) = g * |x-y|^2

#   - if K == D , gamma is a diagonal matrix:
#                 WeightedSquareNorm( g, x-y ) = < x-y, diag(g) * (x-y) >
#                                              = \sum_d  ( g[d] * ((x-y)[d])**2 )
#   - if K == D*D, gamma is a (symmetric) matrix:
#                 WeightedSquareNorm( g, x-y ) = < x-y, g * (x-y) >
#                                              = \sum_{k,l}  ( g[k,l] * (x-y)[k]*(x-y)[l] )
#
# N.B.: Beware of Shape([D]) != Shape([1,D]) confusions !

# Isotropic, uniform kernel -----------------------------------------------------------
sigma = torch.tensor( [0.1] ).type(dtype)
params["gamma"] = 1./sigma**2
showcase_params(params, "Isotropic Uniform kernel", 1)

# Isotropic, Variable kernel ----------------------------------------------------------
sigma = torch.tensor( [ 
    [0.15], 
    [0.07], 
    [0.3] 
    ]).type(dtype)
params["gamma"] = 1./sigma**2
showcase_params(params, "Isotropic Variable kernel", 4)

# Diagonal, Uniform kernel ---------------------------------------------------------
sigma = torch.tensor( [0.2, 0.1] ).type(dtype)
params["gamma"] = 1./sigma**2
showcase_params(params, "Diagonal Uniform kernel", 2)

# Diagonal, Variable kernel --------------------------------------------------------
sigma = torch.tensor( [ 
    [0.2, 0.1], 
    [.05, .15], 
    [.2,  .2] 
    ] ).type(dtype)
params["gamma"] = 1./sigma**2
showcase_params(params, "Diagonal Variable kernel", 5)

# Fully-Anisotropic, Uniform kernel ---------------------------------------------------
Sigma = torch.tensor( [1/0.2**2, 1/.25**2, 1/.25**2, 1/0.1**2 ] ).type(dtype)
params["gamma"]   = Sigma
#params["backend"] = "pytorch"
showcase_params(params, "Fully-Anisotropic Uniform kernel", 3)

# Fully-Anisotropic, Variable kernel --------------------------------------------------
Sigma = torch.tensor( [ 
    [1/0.2**2, 1/.25**2, 1/.25**2, 1/0.1**2  ] ,
    [1/0.1**2,     0,       0,     1/0.12**2 ] ,
    [1/0.3**2,-1/.25**2,-1/.25**2, 1/0.12**2 ] ,
    ] ).type(dtype)
params["gamma"] = Sigma
showcase_params(params, "Fully-Anisotropic Variable kernel", 6)

plt.gcf().set_size_inches(18,12)

import os
fname = "output/anisotropic_kernels.png"
os.makedirs(os.path.dirname(fname), exist_ok=True)
plt.savefig( fname, bbox_inches='tight' )


print("Done. Close the figure to exit.")
plt.show(block=True)

