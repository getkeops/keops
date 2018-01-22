
import torch
from   torch.autograd import Variable
import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) \
				+os.path.sep+'..'+os.path.sep+'..'+os.path.sep+'..'+os.path.sep+'..')
from libkp.torch.kernels import StandardKernelProduct



def _kernel_product(x,y,b, params, mode = "sum") :
	"""
	Simple wrapper around the convenient 'StandardKernelProduct' libkp routine.
	Feel free to implement your own (non-standard) ideas here!
	"""
	name    = params["name"]
	backend = params.get("backend", "auto")
	# gamma should have been generated along the lines of "Variable(torch.Tensor([1/(s**2)])).type(dtype)"
	gamma   = params["gamma"]

	return StandardKernelProduct(gamma, x,y,b, name, mode, backend)

