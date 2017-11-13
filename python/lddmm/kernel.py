# Import the relevant tools
import numpy as np          # standard array library
import torch
from torch.autograd import Variable

from examples.kernel_product import KernelProduct

BACKEND = "libkp" # or "pytorch"

if   BACKEND == "pytorch" :
	def _kernelproduct(s, x, y, p, mode) :
		return _k(x, y, s, mode) @ p
	
elif BACKEND == "libkp" :
	__kernelproduct = KernelProduct().apply
	def _kernelproduct(s, x, y, p, mode) :
		return __kernelproduct(s, x, y, p, mode)

def _squared_distances(x, y) :
	"Returns the matrix of $\|x_i-y_j\|^2$."
	x_col = x.unsqueeze(1) # Theano : x.dimshuffle(0, 'x', 1)
	y_lin = y.unsqueeze(0) # Theano : y.dimshuffle('x', 0, 1)
	return torch.sum( (x_col - y_lin)**2 , 2 )

def _k(x, y, s, mode) :
	sq = _squared_distances(x, y) 
	if   mode == "gaussian" :
		K = torch.exp( -sq / (s**2))
		#K = torch.exp( -torch.sqrt(sq+.0001) )
		#K = torch.pow( 1. / ( 1. + sq ), .25 )
	elif mode == "laplacian" :
		K = torch.exp( -torch.sqrt(s**2 + sq))
	elif mode == "energy" :
		K = torch.pow( 1. / ( s**2 + sq ), .25 )
	return K

def _cross_kernels(q, x, s) :
	"Returns the full k-correlation matrices between two point clouds q and x."
	K_qq = _k(q, q, s)
	K_qx = _k(q, x, s)
	K_xx = _k(x, x, s)
	return (K_qq, K_qx, K_xx)
