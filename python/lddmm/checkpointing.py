import numpy as np          # standard array library
import torch
from   torch.autograd import Variable

from visualize import make_dot


class Checkpoint(torch.autograd.Function) :
	
	def __init__(self, fun, nits, save_every_N = 10) :
		self.fun          = fun
		self.nits         = nits
		self.save_every_N = save_every_N
	
	def forward(self, *args) :
		"""
		Iterate self.fun, for self.nits iteration.
		Computations are made in volatile mode, as we only store one result
		per self.save_every_N iterations.
		"""
		# N.B. : Remember that Pytorch unwraps "args" from Variable to Tensor !
		saved_x = [] ; stepsizes = []
		x       = ( Variable(arg, volatile=False) for arg in args ) # Forget the history
		
		stepsize = 0
		for it in range(self.nits) :
			if it % self.save_every_N == 0 :
				saved_x.append(x)
				stepsizes.append(stepsize) ; stepsize = 0
			x = self.fun(self, *x) ; stepsize += 1
		stepsizes.append(stepsize)
		
		self.saved_x = saved_x ; self.stepsizes = stepsizes[1:]
		return tuple( a.data for a in x )
		
	def backward(self, *grad_args):
		"""
		Back-propagates the gradient, recomputing f^n(x) as needed.
		"""
		grad = grad_args
		
		# Let's climb back the ladder:
		for x_s, stepsize in reversed(list(zip(self.saved_x, self.stepsizes))) :
			print(stepsize)
			x_c = [Variable(xs.data, requires_grad = True) for xs in x_s]
			x   = x_c
			for it in range(stepsize) :
				x = self.fun(self, *x)
			print('x : ', x)
			print('xc : ', x_c)
			make_dot(x[1], params={'a': x_c[0],'b': x_c[1], 'P':self.P, 'Q':self.Q, 'K':self.K}).view()
			# At this point, x = f^{(stepsize)} (x_c)
			grad = torch.autograd.grad( x, x_c, grad )
		
		return grad












