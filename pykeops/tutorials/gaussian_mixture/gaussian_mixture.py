import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + (os.path.sep + '..')*3)

# Standard imports
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import torch
from torch.autograd import grad
from torch.nn       import Module, Parameter
from torch.nn.functional import softmax, log_softmax
from pykeops.torch  import Kernel, kernel_product

plt.ion()

# Choose the storage place for our data : CPU (host) or GPU (device) memory.
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


# Define our Dataset =====================================================================
N = 500
t = torch.linspace(0, 2*np.pi, N+1)[:-1]
x = torch.stack( (.5 + .4*(t/7)*t.cos(), .5+.3*t.sin()), 1)
x = x + .02*torch.randn(x.shape)
x = x.type(dtype)
x.requires_grad = True


# Display ================================================================================
# Create a uniform grid on the unit square:
res = 200
ticks  = np.linspace( 0, 1, res+1)[:-1] + .5 / res 
X,Y    = np.meshgrid( ticks, ticks )

grid = torch.from_numpy(np.vstack( (X.ravel(), Y.ravel()) ).T).contiguous().type(dtype)

# Define our Gaussian Mixture Model =======================================================

class GaussianMixture(Module) :
    def __init__(self, M, sparsity = 0, D = 2) :
        super(GaussianMixture, self).__init__()

        # Let's use a mixture of "gaussian" kernels, i.e.
        #        k(x_i,y_j) = exp( - WeightedSquaredNorm(gamma, x_i-y_j ) )
        self.params    = { "id"     : Kernel("gaussian(x,y)")  }
        self.mu        = Parameter( torch.rand(  M,D   ).type(dtype) )
        self.A         = 10* torch.ones(M,1,1) * torch.eye(D,D).view(1,D,D)
        self.A         = Parameter( ( self.A ).type(dtype).contiguous() )
        self.w         = Parameter( torch.ones(  M,1   ).type(dtype) )
        self.sparsity  = sparsity

    def update_covariances(self) :
        """Computes the full covariance matrices from the model's parameters."""
        (M,D,_) = self.A.shape
        self.params["gamma"] = (self.A @ self.A.transpose(1,2)).view(M, D*D)

    def covariances_determinants(self) :
        """Computes the determinants of the covariance matrices."""
        S = self.params["gamma"]
        if S.shape[1] == 2*2: dets = S[:,0]*S[:,3] - S[:,1]*S[:,2]
        else :                raise NotImplementedError
        return dets.view(-1,1)

    def weights(self) :
        """Scalar factor in front of the exponential, in the density formula."""
        return     softmax(self.w, 0) * self.covariances_determinants().sqrt()

    def weights_log(self) :
        """Logarithm of the scalar factor, in front of the exponential."""
        return log_softmax(self.w, 0) + .5 * self.covariances_determinants().log()

    def likelihoods( self, sample) :
        """Samples the density on a given point cloud."""
        self.update_covariances()
        return kernel_product(self.params, sample, self.mu, self.weights(), mode="sum")

    def log_likelihoods( self, sample) :
        """Log-density, sampled on a given point cloud."""
        self.update_covariances()
        return kernel_product(self.params, sample, self.mu, self.weights_log(), mode="lse")

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

       


# Optimization ================================================================================

plt.figure(figsize=(10,10))

model     = GaussianMixture(10, sparsity=200)
optimizer = torch.optim.Adam( model.parameters() )

for it in range(10001) :
    optimizer.zero_grad()             # Reset the gradients (PyTorch syntax...).
    cost = model.neglog_likelihood(x) # Cost to minimize
    cost.backward()                   # Backpropagate to compute the gradient.
    optimizer.step()

    if it % 10  == 0 : print("Iteration ",it,", Cost = ", cost.data.cpu().numpy())
    if it % 500 == 0 :       
        plt.gcf()
        model.plot(x)
        plt.title("Density, iteration "+str(it), fontsize=20)
        plt.pause(.2)
        
import os
fname = "output/gaussian_mixture.png"
os.makedirs(os.path.dirname(fname), exist_ok=True)
plt.savefig( fname, bbox_inches='tight' )

print("Done. Close the figure to exit.")
plt.show(block=True)
