import torch         # GPU + autodiff library
from visualize import make_dot # GraphViz tool to plot graphs
# See github.com/szagoruyko/functional-zoo/blob/master/visualize.py

# With PyTorch, using the GPU is that simple:
use_gpu  = torch.cuda.is_available()
dtype    = torch.cuda.FloatTensor if use_gpu else torch.FloatTensor
# Under the hood, this flag will determine the backend used for
# forward and backward operations, as they have all been 
# implemented both in pure CPU and in GPU (CUDA) code.

N = 1000; D = 3 ; # Work with clouds of 10,000 points in 3D
# Generate arbitrary arrays on the host (CPU) or device (GPU):
q = torch.linspace( 0, 5, N*D ).type(dtype).view(N,D)
p = torch.linspace( 3, 6, N*D ).type(dtype).view(N,D)
s = torch.Tensor(     [2.5]   ).type(dtype)

# Wrap them into "autodiff" graph nodes. In this demo, 
# we won't try to fine tune the deformation model, so 
# we do not need any derivative with respect to s:
q = torch.autograd.Variable( q, requires_grad = True )
p = torch.autograd.Variable( p, requires_grad = True )
s = torch.autograd.Variable( s, requires_grad = False)

# Actual computations. Every PyTorch instruction is executed
# on-the-fly, but the graph API 'torch.autograd' keeps track of
# the order of the operations and stores in memory the intermediate
# results that are needed for the backward pass.
q_i  = q.unsqueeze(1) # shape (N,D) -> (N,1,D)
q_j  = q.unsqueeze(0) # shape (N,D) -> (1,N,D)
sqd  = torch.sum( (q_i - q_j)**2 , 2 ) # |q_i-q_j|^2
K_qq = torch.exp( - sqd / (s**2) )     # Gaussian kernel
v    = K_qq @ p # matrix multiplication. (N,N)@(N,D) = (N,D)
# Finally, compute the Hamiltonian H(q,p):
H    = .5 * torch.dot( p.view(-1), v.view(-1) ) # .5*<p,v>

gp   = torch.autograd.grad(H, p, create_graph=True)[0]


# Display -- see next figure.
print(H); make_dot(H, {'$q$':q, '$p$':p, '$s$':s}, 
                   stored_vars = ['$q_i-q_j$', '$-\|q_i-q_j\|^2$', '$s^2$', 
                                  '$-|q_i-q_j|^2/s^2$', '$K_{q,q}$', '$p$', 
                                   '$K_{q,q}p$', '$p$'], mode = 'latex'
                   ).render('hamiltonian_kernel.dot', view=True)
# dot2tex --figonly -f tikz --autosize -t raw hamiltonian_kernel.dot > hamiltonian_kernel.tex
                   
#print(gp); make_dot(gp, {'$q$':q, '$p$':p, '$s$':s}, 
#                   stored_vars = [], mode = 'pdf'
#                   ).render('hamiltonian_kernel_gp.dot', view=True)
