import torch
from torch.autograd import grad
from pykeops.torch  import Kernel, kernel_product

import numpy as np
from matplotlib import pyplot as plt
from scipy import misc
from time  import time

tensor    = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor 

# ================================================================================================
# ================================  The Sinkhorn algorithm  ======================================
# ================================================================================================

# Parameters of our optimal transport cost -------------------------------------------------------
a = 1 # Use a cost function  "C(x,y) = |x-y|^a"
# The Sinkhorn algorithm relies on a kernel "k(x,y) = exp(-C(x,y))" :
kernel_names = { 1 : "laplacian" ,   # exp(-|x|),   Earth-mover distance
                 2 : "gaussian"    } # exp(-|x|^2), Quadratic "Wasserstein" distance
params = {
    "id"          : Kernel( kernel_names[a] + "(x,y)" ),

    # Default set of parameters for the Sinkhorn cost - they make sense on the unit square/cube:
    "nits"        : 30,                 # Number of iterations in the Sinkhorn loop
    "epsilon"     : tensor([ .05**a ]), # Regularization strength, homogeneous to C(x,y)
}

# The Sinkhorn loop ------------------------------------------------------------------------------
dot = lambda a,b : torch.dot(a.view(-1), b.view(-1))
def OT_distance(params, Mu, Nu) :
    """
    Computes an optimal transport cost using the Sinkhorn algorithm, stabilized in the log-domain.
    See the section 4.4 of
       "Gabriel Peyré and Marco Cuturi, Computational Optimal Transport, ArXiv:1803.00567, 2018"
    for reference.
    """
    # Instead of "gamma" or "sigma", the Sinkhorn algorithm is best understood in terms
    # of a regularization strength "epsilon", which divides the cost.
    eps = params["epsilon"]
    # The kernel_product convention is that gamma is the *squared distance multiplier*
    # (just like the \Sigma/2 of multivariate Gaussian laws)
    params["gamma"] = 1 / eps**(2/a)

    mu_i, x_i = Mu ; nu_j, y_j = Nu
    mu_i,   nu_j    = mu_i.view(-1,1), nu_j.view(-1,1)
    mu_log, nu_log  = mu_i.log(),      nu_j.log()

    # Initialize the dual variables to zero:
    U, V = torch.zeros_like(mu_log), torch.zeros_like(nu_log)

    # The Sinkhorn loop... is best implemented in the log-domain ! ----------------------------
    for it in range(params["nits"]) :
        # Kernel products + pointwise divisions, in the log-domain.
        # Mathematically speaking, we're alternating Kullback-Leibler projections.
        # N.B.: By convention, U is the deformable source and V is the fixed target. 
        #       If we break before convergence, it is thus important to finish
        #       with a "projection on the mu-constraint"!
        V = - kernel_product(params, y_j, x_i, mu_log+U, mode="lse" ) 
        U = - kernel_product(params, x_i, y_j, nu_log+V, mode="lse" )

    # To compute the full mass of the regularized transport plan (used in the corrective term), 
    # we use a "bonus" kernel_product mode, "log_scaled". Using generic_sum would have been possible too.
    Gamma1 =  kernel_product(params, x_i, y_j, torch.ones_like(V), mu_log+U, nu_log+V, mode="log_scaled")

    # The Sinkhorn cost is homogeneous to C(x,y)
    return eps * ( dot(mu_i, U) + dot(nu_j, V) - dot( Gamma1, torch.ones_like(Gamma1) ) )




# ================================================================================================
# =====================   .png <-> Point clouds conversion routines    ===========================
# ================================================================================================

def LoadImage(fname) :
    img = misc.imread(fname, flatten = True) # Grayscale
    img = (img[::-1, :])  / 255. # [0,1] range
    return tensor( 1 - img )     # Black = 1, White = no mass

def extract_point_cloud(I, scaling = 1.) :
    """Bitmap array to point cloud."""
    ind  = (I > .01).nonzero()   # Threshold, to extract the relevant indices
    mu_i = I[ind[:,0], ind[:,1]] # Extract the weights
    x_i  = ind.float() / scaling # Normalize the point cloud spread
    mu_i = mu_i / mu_i.sum()     # For the sake of simplicity, let's normalize the measures involved

    return ind, mu_i, x_i

def sparse_distance_bmp(params, A, B) :
    """
    Takes as input two torch bitmaps (2D Tensors). 
    Returns a cost and a gradient (encoded as a vector bitmap for display).
    """
    scaling = A.shape[0] # The dataset is rescaled to fit into the unit square

    ind_A, mu_i, x_i = extract_point_cloud(A, scaling)
    ind_B, nu_j, y_j = extract_point_cloud(B, scaling)

    x_i.requires_grad = True # We'll plot the gradient wrt. the x_i's

    # Compute the distance between the *measures* A and B ------------------------------
    print("{:,}-by-{:,} KP... ".format(len(x_i), len(y_j)), end='')

    cost = OT_distance( params, (mu_i,x_i), (nu_j,y_j) )
    grad_x = grad( cost, [x_i] )[0].data # gradient wrt the voxels' positions

    # Point cloud to bitmap (grad_x) ---------------------------------------------------
    grad_A = torch.zeros( *(tuple(A.shape) + (2,)), dtype=A.dtype, device=A.device )
    grad_A[ind_A[:,0],ind_A[:,1],:] = grad_x[:,:] 

    # N.B.: we return "PLUS gradient", i.e. "MINUS a descent direction".
    return cost, grad_A




# ================================================================================================
# ======================================   Demo    ===============================================
# ================================================================================================


source = LoadImage("data/amoeba_1.png")
target = LoadImage("data/amoeba_2.png")

plt.figure(figsize=(10,10))

print("Libraries+Data loaded.")
t_0 = time()
cost, grad_src = sparse_distance_bmp(params, source, target )
t_1 = time()
print("Nits = {:2d}, {:.2f}s, cost : {:.6f}".format(params["nits"], t_1-t_0, cost.data.cpu().item() ))



# Display ==================================================================================
grad_src = - grad_src # N.B.: We want to visualize a descent direction, not the opposite!

# Source + Target :
source_plot = .5*source.cpu().numpy()
target_plot = .5*target.cpu().numpy()
img_plot    = np.dstack( ( np.ones(source_plot.shape) - target_plot, 
                           np.ones(source_plot.shape) - source_plot - target_plot, 
                           np.ones(source_plot.shape) - source_plot ) )

plt.imshow( img_plot, origin="lower", extent=(0,1,0,1))

# Subsample the gradient field :
sample = 4
grad_plot = grad_src.cpu().numpy()
grad_plot = grad_plot[::sample, ::sample, :]

X,Y   = np.meshgrid( np.linspace(0, 1, grad_plot.shape[0]+1)[:-1] + .5/(sample*grad_plot.shape[0]), 
                     np.linspace(0, 1, grad_plot.shape[1]+1)[:-1] + .5/(sample*grad_plot.shape[1]) )

scale = (grad_src[:,:,1]**2 + grad_src[:,:,0]**2).sqrt().mean().item()
plt.quiver( X, Y, grad_plot[:,:,1], grad_plot[:,:,0], 
            scale = .25*scale, scale_units="dots", color="#438B2A")


import os
fname = "output/optimal_transport.png"
os.makedirs(os.path.dirname(fname), exist_ok=True)
plt.savefig( fname, bbox_inches='tight' )

plt.show()
