import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '..')

from examples.kernel_product import KernelProduct


import torch         # GPU + autodiff library
from visualize import make_dot # GraphViz tool to plot graphs
# See github.com/szagoruyko/functional-zoo/blob/master/visualize.py

# With PyTorch, using the GPU is that simple:
use_gpu  = False # torch.cuda.is_available()
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

kernelproduct = KernelProduct.apply
v    = kernelproduct(s, q, q, p, "gaussian") # matrix multiplication. (N,N)@(N,D) = (N,D)
# Finally, compute the Hamiltonian H(q,p):
H    = .5 * torch.dot( p.view(-1), v.view(-1) ) # .5*<p,v>



# Display -- see next figure.
print(H); make_dot(H, {'$q$':q, '$p$':p, '$s$':s}, 
                   stored_vars = ['$s$', '$q$', '$q$', '$p$',
                                  '$K_{q,q}$', '$p$'], mode = 'latex'
                   ).render('hamiltonian_KP.dot', view=True)
# dot2tex --figonly -f tikz --autosize -t raw hamiltonian_kernel.dot > hamiltonian_kernel.tex



# ============================================================
# Just KernelProd :

N = 1000; M = 2000; D = 3 ; E = 1;
# Generate arbitrary arrays on the host (CPU) or device (GPU):
x = torch.linspace( 0, 5, N*D ).type(dtype).view(N,D)
y = torch.linspace( 0, 5, M*D ).type(dtype).view(M,D)
b = torch.linspace( 3, 6, M*E ).type(dtype).view(M,E)
s = torch.Tensor(     [2.5]   ).type(dtype)

# Wrap them into "autodiff" graph nodes. In this demo, 
# we won't try to fine tune the deformation model, so 
# we do not need any derivative with respect to s:
x = torch.autograd.Variable( x, requires_grad = True )
y = torch.autograd.Variable( y, requires_grad = True )
b = torch.autograd.Variable( b, requires_grad = True )
s = torch.autograd.Variable( s, requires_grad = False)

kernelproduct = KernelProduct.apply
v    = kernelproduct(s, x, y, b, "gaussian") # matrix multiplication. (N,M)@(M,E) = (N,E)

a = v
                   
v_x = torch.autograd.grad(v, x, a, create_graph=True)[0]  
v_y = torch.autograd.grad(v, y, a, create_graph=True)[0]  
v_b = torch.autograd.grad(v, b, a, create_graph=True)[0]  

make_dot(v, {'$x$':x, '$y$':y, '$b$':b, '$s$':s}, 
       stored_vars = ['$s$', '$x$','$y$','$b$'], mode = 'latex'
       ).render('v.dot', view=True)                 
make_dot(v_x, {'$x$':x, '$y$':y, '$b$':b, '$s$':s}, 
       stored_vars = ['$s$', '$x$','$y$','$b$','$s$','$a$','$x$','$y$', '$b$'], mode = 'latex'
       ).render('v_x.dot', view=True)                 
make_dot(v_y, {'$x$':x, '$y$':y, '$b$':b, '$s$':s}, 
       stored_vars = ['$s$', '$x$','$y$','$b$','$s$','$a$','$x$','$y$', '$b$'], mode = 'latex'
       ).render('v_y.dot', view=True)                 
make_dot(v_b, {'$x$':x, '$y$':y, '$b$':b, '$s$':s}, 
       stored_vars = ['$s$', '$x$','$y$','$b$','$s$','$a$','$x$','$y$', '$b$'], mode = 'latex'
       ).render('v_b.dot', view=True)    
                   
