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
from torch.nn       import Module, Parameter
from torch.nn.functional import softmax, log_softmax
from pykeops.torch.kernels import Kernel, kernel_product

plt.ion()

# Choose the storage place for our data : CPU (host) or GPU (device) memory.
use_cuda = torch.cuda.is_available()
dtype    = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor


# Define our Dataset =====================================================================
N = 500
t = torch.linspace(0, 2*np.pi, N+1)[:-1]
x = torch.stack( (.5 + .4*(t/7)*t.cos(), .5+.3*t.sin()), 1)
x = x + .02*torch.randn(x.shape)
x = Variable(x, requires_grad=True).type(dtype)


# Display ================================================================================
# Create a uniform grid on the unit square:
res = 200
ticks  = np.linspace( 0, 1, res+1)[:-1] + .5 / res 
X,Y    = np.meshgrid( ticks, ticks )

grid = Variable(torch.from_numpy(np.vstack( (X.ravel(), Y.ravel()) ).T).contiguous().type(dtype) )

# Define our Gaussian Mixture Model =======================================================

class GaussianMixture(Module) :
    def __init__(self, M, sparsity = 0) :
        super(GaussianMixture, self).__init__()

        # Let's use a mixture of "gaussian" kernels, i.e.
        #        k(x_i,y_j) = exp( - WeightedSquaredNorm(gamma, x_i-y_j ) )
        self.params    = { "id"     : Kernel("gaussian(x,y)") , "backend" : "pytorch" }
        self.mu        = Parameter(   torch.rand(  M,2 ).type(dtype)   )
        self.sigma_log = Parameter(-2*torch.ones(  M,2 ).type(dtype)   )
        self.theta     = Parameter(   torch.zeros( M   ).type(dtype)   )
        self.w         = Parameter(   torch.ones(  M,1 ).type(dtype)/M )
        self.sparsity  = sparsity

    def update_covariances(self) :
        """Computes the full covariance matrices from the model's parameters."""
        sig2 = (-2*self.sigma_log).exp()
        a, b     = sig2[:,0], sig2[:,1]
        cos, sin = self.theta.cos(), self.theta.sin()
        self.params["gamma"] = torch.stack( (
            a * cos**2 + b * sin**2 ,
            (a-b) * cos * sin,
            (a-b) * cos * sin,
            a * sin**2 + b * cos**2
        ), 1)

    def weights(self) :
        """Scalar factor in front of the exponential, in the density formula."""
        return     softmax(self.w, 0) * (-self.sigma_log.sum(1).view(-1,1)).exp()

    def weights_log(self) :
        """Logarithm of the scalar factor, in front of the exponential."""
        return log_softmax(self.w, 0) - self.sigma_log.sum(1).view(-1,1)

    def likelihoods( self, sample) :
        """Samples the density on a given point cloud."""
        self.update_covariances()
        return kernel_product( sample, self.mu, self.weights(), self.params)

    def log_likelihoods( self, sample) :
        """Log-density, sampled on a given point cloud."""
        self.update_covariances()
        return kernel_product( sample, self.mu, self.weights_log(), self.params, mode="log")

    def neglog_likelihood( self, sample ) :
        """Returns -log(likelihood(sample)) up to an additive factor."""
        ll = self.log_likelihoods(sample)
        log_likelihood  = torch.dot( ll.view(-1) , torch.ones_like(ll).view(-1) )
        return -log_likelihood + self.sparsity * softmax(self.w, 0).sqrt().sum()

    def get_sample(self, N) :
        """Generates a sample of N points."""
        raise NotImplementedError()

    def plot(self, sample) :
        """Displays the model."""
        plt.clf()
        # Heatmap
        heatmap   = self.likelihoods( grid )
        heatmap   = heatmap.view(res,res).data.cpu().numpy() # reshape as a "background" image

        scale = np.amax( np.abs( heatmap[:]) )
        plt.imshow(  -heatmap, interpolation='bilinear', origin='lower', 
                    vmin = -scale, vmax = scale, cmap=cm.RdBu, 
                    extent=(0,1,0,1)) 

        # Log-contours
        log_heatmap   = self.log_likelihoods( grid )
        log_heatmap   = log_heatmap.view(res,res).data.cpu().numpy()

        scale = np.amax( np.abs( log_heatmap[:]) )
        levels = np.linspace(-scale, scale, 41)

        plt.contour(log_heatmap, origin='lower', linewidths = 1., colors = "#C8A1A1",
                    levels = levels, extent=(0,1,0,1)) 

        # Dataset scatter plot
        xy = sample.data.cpu().numpy()
        plt.scatter( xy[:,0], xy[:,1], 4, color='k' )

        # Title
        plt.title("Density", fontsize=20)


# Optimization ================================================================================

plt.figure(figsize=(10,10))

model     = GaussianMixture(8, sparsity=500)
optimizer = torch.optim.Adam( model.parameters() )

for it in range(10001) :
    optimizer.zero_grad()             # Reset the gradients (PyTorch syntax...).
    cost = model.neglog_likelihood(x) # Cost to minimize
    cost.backward()                   # Backpropagate to compute the gradient.
    optimizer.step()

    if it % 10  == 0 : print("Iteration ",it,", Cost = ", cost.data.cpu().numpy()[0])
    if it % 500 == 0 :
        model.plot(x)
        plt.pause(.1)

print("Done.")
plt.show(block=True)
