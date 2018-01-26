
import torch
from   torch.autograd import Variable
import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) \
				+(os.path.sep+'..')*5)
from libkp.torch.kernels import KernelProduct, Kernel



def _kernel_product(x,y,b, params, mode = "sum", bonus_args = None) :
	"""
	Simple wrapper around the convenient 'KernelProduct' libkp routine.
	Feel free to implement your own (non-standard) ideas here!
	"""
	kernel  = params["id"]
	backend = params.get("backend", "auto")
	# gamma should have been generated along the lines of "Variable(torch.Tensor([1/(s**2)])).type(dtype)"
	gamma   = params["gamma"]
	
	return KernelProduct(gamma, x,y,b, kernel, mode, backend, bonus_args = bonus_args)

