import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '..')


# Import the relevant tools
import numpy as np          # standard array library
import torch

# No need for a ~/.theanorc file anymore !
use_cuda = False #torch.cuda.is_available()
dtype    = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
dtypeint = torch.cuda.LongTensor  if use_cuda else torch.LongTensor

from kernel import _kernelproduct

NSTEPS    = 10
DT        = 1 / NSTEPS
MODE      = "energy"


# Pytorch is a fantastic deep learning library : it transforms symbolic python code
# into highly optimized CPU/GPU binaries, which are called in the background seamlessly.
# It can be thought of as a "heir" to the legacy Theano library (RIP :'-( ):
# As you'll see, migrating a codebase from one to another is fairly simple !
#
# N.B. : On my Dell laptop, I have a GeForce GTX 960M with 640 Cuda cores and 2Gb of memory.
#
# We now show how to code a whole LDDMM pipeline into one (!!!) page of torch symbolic code.

# Part 1 : cometric on the space of landmarks, kinetic energy on the phase space (Hamiltonian)===



def _integrate_euler(X, flow, traj = False) :
	if traj : Xs = [X]
	
	# Summing lists in python is a bit convoluted...
	for t in range(NSTEPS) :
		X = [x + DT * dx for (x, dx) in zip(X, flow(*X))] 
		if traj : Xs.append(X)
	
	if traj :
		return list(zip(*Xs))
	else :
		return X

def _integrate_rk4(X, flow, traj = False) :
	if traj : Xs = [X]
		
	# Summing lists in python is a bit convoluted...
	for t in range(NSTEPS) :
		K1 = flow(*X)
		K2 = flow(*[ x + .5*DT*k1 for (x,k1) in zip(X, K1)])
		K3 = flow(*[ x + .5*DT*k2 for (x,k2) in zip(X, K2)])
		K4 = flow(*[ x +    DT*k3 for (x,k3) in zip(X, K3)])
		X = [x + (DT/6) * (k1 + 2*k2 + 2*k3 + k4) 
		     for (x, k1, k2, k3, k4) in zip(X, K1, K2, K3, K4)] 
		if traj : Xs.append(X)
	
	if traj :
		return list(zip(*Xs))
	else :
		return X

def _integrate(*args, **argv) :
	return _integrate_euler(*args, **argv)

def _Hqp(q, p, sigma, q_mu = None) :
	"""
	The hamiltonian, or kinetic energy of the shape q with momenta p.
	NB: the q_mu is a feature I'm working on with Alain Trouvé, not public yet.
	"""
	pKqp =  torch.dot(p.view(-1), _kernelproduct(sigma, q, q, p, MODE).view(-1))
	return .5 * pKqp.sum()                # $H(q,p) = \frac{1}{2} * sum_{i,j} k(x_i,x_j) p_i.p_j$
    
# Part 2 : Geodesic shooting ====================================================================
# The partial derivatives of the Hamiltonian are automatically computed !
def _dq_Hqp(q,p,sigma, q_mu=None) : 
	return torch.autograd.grad(_Hqp(q,p,sigma, q_mu = q_mu), q, create_graph=True)[0]
def _dp_Hqp(q,p,sigma, q_mu=None) :
	return torch.autograd.grad(_Hqp(q,p,sigma, q_mu = q_mu), p, create_graph=True)[0]

def _hamiltonian_flow(sigma, q_mu = None) :
	def flow(q, p) :
		return [   _dp_Hqp(q,p,sigma, q_mu = q_mu) , 
				 - _dq_Hqp(q,p,sigma, q_mu = q_mu) ]
	return flow

def _hamiltonian_carrying_flow(sigma, q_mu = None) :
	def flow(q, p, g) :
		return [   _dp_Hqp(q,p, sigma, q_mu = q_mu), 
				 - _dq_Hqp(q,p, sigma, q_mu = q_mu), 
				   _kernelproduct(sigma, g, q, p, MODE) ]
	return flow

def _HamiltonianShooting(q, p, sigma, q_mu = None) :
	"Shoots to time 1 a k-geodesic starting (at time 0) from q with momentum p."
	return _integrate( [q,p], _hamiltonian_flow(sigma, q_mu = q_mu) )                       
	
# Part 2bis : Geodesic shooting + deformation of the ambient space, for visualization ===========
def _HamiltonianCarrying(q, p, g, sigma, q_mu = None) :
	"""
	Similar to _HamiltonianShooting, but also conveys information about the deformation of
	an arbitrary point cloud 'grid' in the ambient space.
	""" 
	return _integrate( [q,p,g], _hamiltonian_carrying_flow(sigma, q_mu = q_mu) )   
	

def _HamiltonianTrajectory(q, p, g, sigma, q_mu = None) :
	"""
	Similar to _HamiltonianCarrying, but returns the whole trajectory.
	"""
	return _integrate( [q,p,g], _hamiltonian_carrying_flow(sigma, q_mu=q_mu), traj = True ) 
















