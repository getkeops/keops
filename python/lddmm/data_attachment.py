# Import the relevant tools
import time                 # to measure performance
import numpy as np          # standard array library
import torch
from   torch.autograd import Variable
import torch.optim as optim

# No need for a ~/.theanorc file anymore !
use_cuda = False #torch.cuda.is_available()
dtype    = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
dtypeint = torch.cuda.LongTensor  if use_cuda else torch.LongTensor

from kernel import _k, _cross_kernels, _squared_distances

# Part 3 : Data attachment ======================================================================

def _squared_euclidean(q1_x, q1_mu, xt_x, xt_mu) :
	distances = ((q1_x - xt_x)**2).sum(1)
	mean_L2 = .5 * ( (q1_mu+xt_mu) * distances ).sum(0)
	return mean_L2, mean_L2 # We have to give something as info...

def _kernel_matching(q1_x, q1_mu, xt_x, xt_mu, radius) :
	"""
	Given two measures q1 and xt represented by locations/weights arrays, 
	outputs a kernel-fidelity term and an empty 'info' array.
	"""
	K_qq, K_qx, K_xx = _cross_kernels(q1_x, xt_x, radius)
	cost = .5 * (   torch.sum(K_qq * torch.ger(q1_mu,q1_mu)) \
				 +  torch.sum(K_xx * torch.ger(xt_mu,xt_mu)) \
				 -2*torch.sum(K_qx * torch.ger(q1_mu,xt_mu))  )
				 
	# Info = the 2D graph of the blurred distance function
	# Increase res if you want to get nice smooth pictures...
	res    = 10 ; ticks = np.linspace( 0, 1, res + 1)[:-1] + 1/(2*res) 
	X,Y    = np.meshgrid( ticks, ticks )
	points = Variable(torch.from_numpy(np.vstack( (X.ravel(), Y.ravel()) ).T).type(dtype), requires_grad=False)
							   
	info   = _k( points, q1_x , radius ) @ q1_mu \
	       - _k( points, xt_x , radius ) @ xt_mu
	return [cost , info.view( (res,res) ) ]
	
def _ot_matching(q1_x, q1_mu, xt_x, xt_mu, radius) :
	"""
	Given two measures q1 and xt represented by locations/weights arrays, 
	outputs an optimal transport fidelity term and the transport plan.
	"""
	# The Sinkhorn algorithm takes as input three Theano variables :
	c = _squared_distances(q1_x, xt_x) # Wasserstein cost function
	mu = q1_mu ; nu = xt_mu
	
	# Parameters of the Sinkhorn algorithm.
	epsilon            = (.08)**2          # regularization parameter
	rho                = (.5) **2          # unbalanced transport (See PhD Th. of Lenaic Chizat)
	niter              = 10000             # max niter in the sinkhorn loop
	tau                = -.8               # nesterov-like acceleration
	
	lam = rho / (rho + epsilon)            # Update exponent
	
	# Elementary operations .....................................................................
	def ave(u,u1) : 
		"Barycenter subroutine, used by kinetic acceleration through extrapolation."
		return tau * u + (1-tau) * u1 
	def M(u,v)  : 
		"$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
		return (-c + u.unsqueeze(1) + v.unsqueeze(0)) / epsilon
	lse = lambda A    : torch.log(torch.exp(A).sum( 1 ) + 1e-6) # slight modif to prevent NaN
	
	# Actual Sinkhorn loop ......................................................................
	u,v,err = 0.*mu, 0.*nu, 0.
	actual_nits = 0
	
	for i in range(niter) :
		u1= u # useful to check the update
		u = ave( u, lam * ( epsilon * ( torch.log(mu) - lse(M(u,v))   ) + u ) )
		v = ave( v, lam * ( epsilon * ( torch.log(nu) - lse(M(u,v).t()) ) + v ) )
		err = (u - u1).abs().sum()
		
		actual_nits += 1
		if (err < 1e-4).data.cpu().numpy() :
			break
	U, V = u, v 
	Gamma = torch.exp( M(U,V) )            # Eventual transport plan g = diag(a)*K*diag(b)
	cost  = torch.sum( Gamma * c )         # Simplistic cost, chosen for readability in this tutorial
	
	print('Sinkhorn error after ' + str(actual_nits) + ' iterations : ' + str(err.data.cpu().numpy()))
	return [cost, Gamma]
	

def _data_attachment(q1_measure, xt_measure, radius, mode = "OT") :
	"Given two measures and a radius, returns a cost - as a Theano symbolic variable."
	if mode == "OT" : # Convenient way to allow the choice of a method
		return _ot_matching(q1_measure[0], q1_measure[1], 
								xt_measure[0], xt_measure[1], 
								radius)
	elif mode == "kernel" :
		return _kernel_matching(q1_measure[0], q1_measure[1], 
								xt_measure[0], xt_measure[1], 
								radius)
	elif mode == "L2" :
		return _squared_euclidean(q1_measure[0], q1_measure[1], 
								xt_measure[0], xt_measure[1])
