import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + (os.path.sep + '..')*2)

# Standard imports
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import torch
from torch          import Tensor
from torch.autograd import Variable, grad
from pykeops.torch.kernels import Kernel, kernel_product

plt.ion()

# Choose the storage place for our data : CPU (host) or GPU (device) memory.
use_cuda = torch.cuda.is_available()
dtype    = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor


# Define our dataset
y = Variable(torch.Tensor( [
    [ .2, .7],
    [ .5, .3],
    [ .7, .5]
    ])).type(dtype)
b = Variable(torch.Tensor([
    1., 1., .5
])).type(dtype)
b = b.view(-1,1) # Remember : b is not a vector, but a "list of unidimensional vectors"!

# Create a uniform grid on the unit square:
res = 100
ticks  = np.linspace( 0, 1, res+1)[:-1] + .5 / res 
X,Y    = np.meshgrid( ticks, ticks )

x = Variable(torch.from_numpy(np.vstack( (X.ravel(), Y.ravel()) ).T).contiguous().type(dtype) )

def showcase_params( params , title) :
    heatmap   = kernel_product(x, y, b, params) # Sample "x -> ∑_j b_j * k_j(x - y_j)" on the grid
    heatmap   = heatmap.view(res,res).data.cpu().numpy() # reshape as a "background" image

    plt.figure(figsize=(10,10))

    plt.imshow(  -heatmap, interpolation='bilinear', origin='lower', 
                vmin = -1, vmax = 1, cmap=cm.RdBu, 
                extent=(0,1,0,1)) 
    plt.title(title, fontsize=20)



# Isotropic, uniform kernel ----------------------------------------------------
sigma = Variable(torch.Tensor( [0.1] )).type(dtype)
isotropic_uniform = {
    "id"      : Kernel("gaussian(x,y)"),
    "gamma"   : 1./sigma**2,
}
showcase_params(isotropic_uniform, "Isotropic Uniform kernel")

# Isotropic, Variable kernel ------------------------------------------------
sigma = Variable(torch.Tensor( [ [0.15], [0.07], [0.3] ] )).type(dtype)
isotropic_variable = {
    "id"      : Kernel("gaussian(x,y)"),
    "gamma"   : 1./sigma**2,
}
showcase_params(isotropic_variable, "Isotropic Variable kernel")

# Anisotropic, Uniform kernel --------------------------------------------------
sigma = Variable(torch.Tensor( [0.2, 0.1] )).type(dtype)
anisotropic_uniform = {
    "id"      : Kernel("gaussian(x,y)"),
    "gamma"   : 1./sigma**2,
}
showcase_params(anisotropic_uniform, "Anisotropic Uniform kernel")

# Anisotropic, Variable kernel --------------------------------------------------
sigma = Variable(torch.Tensor( [ [0.2, 0.1], [.05, .15], [.2, .2] ] )).type(dtype)
anisotropic_uniform = {
    "id"      : Kernel("gaussian(x,y)"),
    "gamma"   : 1./sigma**2,
}
showcase_params(anisotropic_uniform, "Anisotropic Variable kernel")

# Fully-Anisotropic, Uniform kernel --------------------------------------------------
Sigma = Variable(torch.Tensor( [1/0.2**2, 1/.15**2, 1/0.1**2 ] )).type(dtype)
anisotropic_uniform = {
    "id"      : Kernel("gaussian(x,y)"),
    "gamma"   : Sigma,
}
showcase_params(anisotropic_uniform, "Fully-Anisotropic Uniform kernel")

plt.show(block=True)