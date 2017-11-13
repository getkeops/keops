# LDDMM registration using PyTorch
# Author : Jean Feydy

# Choose "libkp" for our CUDA routines, or "pytorch" for the out-of-the-box implementation
BACKEND = "libkp" 

import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../..')


if BACKEND == "libkp" :
	from examples.kernel_product import KernelProduct
	_kernelproduct_raw = KernelProduct().apply

# Note about function names :
#  - MyFunction  is an input-output code
#  - my_routine  is a numpy routine
#  - _my_formula is a PyTorch symbolic function

# Import the relevant tools
import time                 # to measure performance
import numpy as np          # standard array library
import torch
from   torch.autograd import Variable
import torch.optim as optim
from   input_output import level_curves, Curve, GridData, ShowTransport, DisplayShoot, DisplayTrajectory

# Choose the storage place for our data : CPU (host) or GPU (device) memory.
# N.B. : As of Nov. 2017, the "libkp" backend makes computations on the GPU,
#        but requires as input data arrays that are stored on the CPU (host) memory,
#        and load it on the device by itself.
#        Come Jan. 2018, "libkp" should accept float pointers that already live
#        on the GPU.
use_cuda = torch.cuda.is_available() and (BACKEND != "libkp")
dtype    = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
dtypeint = torch.cuda.LongTensor  if use_cuda else torch.LongTensor




# Pytorch is a fantastic deep learning library : it transforms symbolic python code
# into highly optimized CPU/GPU binaries, which are called in the background seamlessly.
# It can be thought of as a "heir" to the legacy Theano library (RIP :'-( ):
# As you'll see, migrating a codebase from one to another is fairly simple !
#
# N.B. : On my Dell laptop, I have a GeForce GTX 960M with 640 Cuda cores and 2Gb of memory.
#
# We now show how to code a whole LDDMM pipeline into one (!!!) page of torch symbolic code.

# Part 1 : cometric on the space of landmarks, kinetic energy on the phase space (Hamiltonian)===

def _squared_distances(x, y) :
	"Returns the matrix of $\|x_i-y_j\|^2$."
	x_col = x.unsqueeze(1) # (N,D) -> (N,1,D)
	y_lin = y.unsqueeze(0) # (M,D) -> (1,M,D)
	return torch.sum( (x_col - y_lin)**2 , 2 )

def _k_raw(x, y, s, mode) :
	sq = _squared_distances(x, y) 
	if   mode == "gaussian"  : return torch.exp( -sq / (s**2))
	elif mode == "laplacian" : return torch.exp( -torch.sqrt(sq+.0001) / s)
	elif mode == "energy"    : return torch.pow( 1. / ( s**2 + sq ), .25 )

if   BACKEND == "pytorch" :
	def _kernelproduct_onestep(s, x, y, p, mode) :
		return _k_raw(x, y, s, mode) @ p
elif BACKEND == "libkp" :
	def _kernelproduct_onestep(s, x, y, p, mode) :
		return _kernelproduct_raw(s, x, y, p, mode)

def _kernelproduct(x, y, p, params, y_mu = None) :
	"""
	If normalize=True, we use a sinkhornized kernel.
	See our report for complete documentation on this new model.
	"""
	s = params["radius"] ; mode = params["mode"] ; normalize = params["normalize"]
	
	if not normalize :
		return _kernelproduct_onestep(s, x, y, p, mode)
	else :
		# We assume that y is the actual measure,
		# and x the point set on which we interpolate
		ly = Variable( torch.ones((y.size(0),1)).type(p.data.type()) )
		
		y_mu = y_mu.view(y.size(0),1) # use pointwise multiplications instead of broadcasting...
		# Sinkhornization loop
		for i in range(5) :
			ly = torch.sqrt( ly / _kernelproduct_onestep(s, y, y, ly * y_mu, mode) )
		return  _kernelproduct_onestep(s, x, y, ly * p,    mode) \
		      / _kernelproduct_onestep(s, x, y, ly * y_mu, mode) 

def _Hqp(q, p, params, q_mu = None) :
	"The hamiltonian, or kinetic energy of the shape q with momenta p."
	pKqp =  torch.dot(p.view(-1), _kernelproduct(q, q, p, params, y_mu = q_mu).view(-1))
	return .5 * pKqp                # $H(q,p) = \frac{1}{2} * sum_{i,j} K_ij p_i.p_j$


# Part 2 : Geodesic shooting ====================================================================
# The partial derivatives of the Hamiltonian are automatically computed !
def _dq_Hqp(q,p, params, q_mu = None) : 
	return torch.autograd.grad(_Hqp(q,p, params, q_mu), q, create_graph=True)[0]

def _dp_Hqp(q,p, params, q_mu = None) :
	return torch.autograd.grad(_Hqp(q,p,params,q_mu), p, create_graph=True)[0]

def _hamiltonian_step(q,p, params, q_mu = None) :
	"Simplistic euler scheme step with dt = .1."
	return [q + .1 * _dp_Hqp(q,p, params,q_mu) ,
			p - .1 * _dq_Hqp(q,p, params,q_mu) ]

def _HamiltonianShooting(q,p, params, q_mu = None) :
	"Shoots to time 1 a k-geodesic starting (at time 0) from q with momentum p."
	for t in range(10) :
		q,p = _hamiltonian_step(q,p, params,q_mu) # Let's hardcode the "dt = .1"
	return [q,p]                             # and only return the final state + momentum



# Part 2bis : Geodesic shooting + deformation of the ambient space, for visualization ===========
def _HamiltonianCarrying(q, p, g, params, q_mu = None, trajectory = False, endtime = 1.) :
	"""
	Similar to _HamiltonianShooting, but also conveys information about the deformation of
	an arbitrary point cloud 'grid' in the ambient space.
	""" 
	if trajectory :
		qs = [q] ; ps = [p]; gs = [g]
	for t in range(int(endtime * 10)) : # Let's hardcode the "dt = .1"
		q,p,g = [q + .1 * _dp_Hqp(          q, p, params, q_mu), 
		         p - .1 * _dq_Hqp(          q, p, params, q_mu), 
		         g + .1 * _kernelproduct(g, q, p, params, q_mu)]
		if trajectory :
			qs.append(q) ; ps.append(p) ; gs.append(g)
	
	if trajectory :
		return qs,ps,gs         # return the states + momenta + grids
	else :
		return q, p, g          # return the final state + momentum + grid

# Part 3 : Data attachment ======================================================================

def _ot_matching(q1_x, q1_mu, xt_x, xt_mu, params) :
	"""
	Given two measures q1 and xt represented by locations/weights arrays, 
	outputs an optimal transport fidelity term and the transport plan.
	"""
	LOG_IMPLEMENTATION = True
	
	# The Sinkhorn algorithm takes as input three variables :
	mu = q1_mu ; nu = xt_mu
	
	# Parameters of the Sinkhorn algorithm.
	epsilon            = params['epsilon']  # regularization parameter
	rho                = params['rho']      # unbalanced transport (See PhD Th. of Lenaic Chizat)
	niter              = params['niter']    # max niter in the sinkhorn loop
	tau                = params['tau']      # nesterov-like acceleration
	
	lam = rho / (rho + epsilon)            # Update exponent
	
	
	if LOG_IMPLEMENTATION : # log domain  implementation, which is more stable numerically,
		                    # but doesn't involve actual convolutions.
		                    # It is not part of libkp just yet.
		c = _squared_distances(q1_x, xt_x) # Wasserstein cost function
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
	else : # Straightforward Sinkhorn algorithm
		mu = mu.view(-1,1) ; nu = nu.view(-1,1)
		a = Variable(torch.ones( mu.size() ).type(dtype) , requires_grad=False)
		b = Variable(torch.ones( nu.size() ).type(dtype) , requires_grad=False)
		err = 0.
		kparams = dict(
			mode      = "gaussian",
			radius    = Variable(torch.from_numpy( np.array([np.sqrt(epsilon)]) ).type(dtype), requires_grad=False),
			normalize = False
		)
		actual_nits = 0
		for i in range(niter) :
			aprev = a
			a = mu / _kernelproduct(q1_x, xt_x, b.view(-1,1), kparams)
			b = nu / _kernelproduct(xt_x, q1_x, a.view(-1,1), kparams)
			err = (a - aprev).abs().sum()
			#print(a,b,err)
			actual_nits += 1
			if (err < 1e-6).data.cpu().numpy() :
				break
		cost = torch.dot( a.view(-1), _kernelproduct(q1_x, xt_x, b.view(-1,1), kparams).view(-1) )
		Gamma = Variable(torch.zeros( (q1_x.size(0),xt_x.size(0)) ).type(dtype))
	print('Sinkhorn error after ' + str(actual_nits) + ' iterations : ' + str(err.data.cpu().numpy()))
	
	return [cost, Gamma]
	
def _kernel_matching(q1_x, q1_mu, xt_x, xt_mu, params) :
	"""
	Given two measures q1 and xt represented by locations/weights arrays, 
	outputs a kernel-fidelity term and an empty 'info' array.
	"""
	kernel_params = dict( radius = params["radius"] , mode = params["mode"], normalize = False )
	def aux( a_m, a_x, b_x, b_m ) :
		return torch.dot( a_m.view(-1) ,
		                  _kernelproduct(a_x, b_x, b_m.view(-1,1), # b_m : (M,) -> (M,1)
		                                 kernel_params, y_mu = None).view(-1) )
	
	cost = .5 * (   aux( q1_mu, q1_x, q1_x, q1_mu) \
				 +  aux( xt_mu, xt_x, xt_x, xt_mu) \
				 -2*aux( q1_mu, q1_x, xt_x, xt_mu ) )
				 
	# Info = the 2D graph of the blurred distance function.
	# To be honest, we should use the convolution with the
	# convolutional square root of the convolution kernel k... 
	# Increase res if you want to get nice smooth pictures.
	res    = 20 ; ticks = np.linspace( 0, 1, res + 1)[:-1] + 1/(2*res) 
	X,Y    = np.meshgrid( ticks, ticks )
	points = Variable(torch.from_numpy(np.vstack( (X.ravel(), Y.ravel()) ).T).type(dtype), requires_grad=False)
	
	info   = _kernelproduct(points, q1_x, q1_mu.view(-1,1), kernel_params, y_mu = None) \
	       - _kernelproduct(points, xt_x, xt_mu.view(-1,1), kernel_params, y_mu = None)
	
	return [cost , info.view( (res,res) ) ]

def _L2_matching(q1_x, q1_mu, xt_x, xt_mu, params) :
	#cost = torch.sum( ( (q1_x - xt_x)**2 ) * q1_mu.view(-1,1) )
	cost = torch.sum( ( (q1_x - xt_x)**2 ))
	return [ cost, cost.view(-1) ]
	


def _data_attachment(q1_measure, xt_measure, params) :
	"Given two measures and a radius, returns a cost - as a Theano symbolic variable."
	if   params["formula"] == "OT" :
		return _ot_matching(q1_measure[0], q1_measure[1], 
							xt_measure[0], xt_measure[1], 
							params)
	elif params["formula"] == "kernel" :
		return _kernel_matching(q1_measure[0], q1_measure[1], 
								xt_measure[0], xt_measure[1], 
								params)
	elif params["formula"] == "L2" :
		return _L2_matching(q1_measure[0], q1_measure[1], 
							xt_measure[0], xt_measure[1], 
							params)

# Part 4 : Cost function and derivatives ========================================================

def _cost( q,p, xt_measure, connec, params, q1_mu = None, q0_mu = None ) :
	"""
	Returns a total cost, sum of a small regularization term and the data attachment.
	.. math ::
	
		C(q_0, p_0) = .01 * H(q0,p0) + 1 * A(q_1, x_t)
	
	Needless to say, the weights can be tuned according to the signal-to-noise ratio.
	"""
	# Geodesic shooting from q0 to q1:
	q1 = _HamiltonianShooting(q,p, params["deformation"], q_mu = q0_mu)[0]  
	# To compute a data attachment cost, we need to turn the set of vertices 'q1' into a measure.
	if   params["transport_action"] == "image"   and params["data_type"] == "curves":
		q1_measure = Curve._vertices_to_measure( q1, connec )
	elif params["transport_action"] == "measure" and params["data_type"] == "curves" :
		q1_measure = Curve._vertices_to_measure( q1, connec )
		q1_measure = (q1_measure[0], q1_mu)
	elif params["data_type"] == "landmarks" :
		q1_measure = (q1, q1_mu)
	else :
		print("This combination has not been implemented.")
	
	attach_and_info = _data_attachment( q1_measure,  xt_measure,  params["data_attachment"] )
	return [   .01* _Hqp(q, p, params["deformation"], q_mu = q0_mu) \
	        + 1.  * attach_and_info[0] , attach_and_info[1] ]

#================================================================================================


def perform_matching( Q0, Xt, params, scale_momentum = 1, scale_attach = 1, show_trajectories = True) :
	"Performs a matching from the source Q0 to the target Xt, returns the optimal momentum P0."
	
	if params["data_type"] == "curves" :
		(Xt_x, Xt_mu) = Xt.to_measure()      # Transform the target into a measure once and for all
		Q0_points   = Q0.points
		Q0_mu       = None      # normalized transport of curves has not been implemented yet
		connec      = torch.from_numpy(Q0.connectivity).type(dtypeint)
		connec_plot = connec
	elif params["data_type"] == "landmarks" :
		(Xt_x, Xt_mu) = Xt
		Q0_points   = Q0[0]
		Q0_mu       = Variable(torch.from_numpy( Q0[1]   ).type(dtype), requires_grad=False)
		connec      = None
		connec_plot = Q0[1] # The weights will be used as a dummy connectivity matrix...
		Q0 = Curve(Q0[0], Q0[1]) # for plotting purpose
		Xt = Curve(Xt_x, Xt_mu)  # for plotting purpose
	
	if   params["transport_action"] == "image" :
		Q1_mu = None
	elif params["transport_action"] == "measure" :
		Q1_mu = Q0_mu
	
	# Declaration -------------------------------------------------------------------------------
	# Cost is a function of 6 parameters :
	# The source 'q',                    the starting momentum 'p',
	# the target points 'xt_x',          the target weights 'xt_mu',
	# the deformation scale 'sigma_def', the attachment scale 'sigma_att'. 
	# N.B. : the .contiguous() is there to ensure that variables will be stored in a contiguous
	#        memory space, which is not always the case if the numpy arrays were constructed
	#        by concatenation operators.
	q0    = Variable(torch.from_numpy(    Q0_points ).type(dtype), requires_grad=True ).contiguous()
	p0    = Variable(torch.from_numpy( 0.*Q0_points ).type(dtype), requires_grad=True ).contiguous()
	Xt_x  = Variable(torch.from_numpy(    Xt_x      ).type(dtype), requires_grad=False).contiguous()
	Xt_mu = Variable(torch.from_numpy(    Xt_mu     ).type(dtype), requires_grad=False).contiguous()
	
	# Encapsulate the float radius. Maybe, one day, we'll implement the derivatives wrt the
	# kernel radii, so just in case...
	params["deformation"    ]["radius"] = \
	Variable(torch.from_numpy( np.array(params["deformation"    ]["radius"]) ).type(dtype), requires_grad=False)
	params["data_attachment"]["radius"] = \
	Variable(torch.from_numpy( np.array(params["data_attachment"]["radius"]) ).type(dtype), requires_grad=False)
	
	print(params["deformation"    ]["radius"])
	
	def Cost(q,p, xt_x,xt_mu, q1_mu, q0_mu) : 
		return _cost( q,p, (xt_x,xt_mu), connec, params, q1_mu, q0_mu )
	
	# Display pre-computing ---------------------------------------------------------------------
	g0,cgrid = GridData() ; G0 = Curve(g0, cgrid )
	g0 = Variable( torch.from_numpy( g0 ).type(dtype), requires_grad = False )
	
	# L-BFGS minimization -----------------------------------------------------------------------
	from scipy.optimize import minimize
	def matching_problem(p0) :
		"""
		Energy minimized in the variable 'p0'.
		q1_mu denotes the weights on the shape after the shooting:
		it will be used iff params["transport action"] == "measure"
		"""
		[c, info] = Cost(q0, p0, Xt_x, Xt_mu, Q1_mu, Q0_mu)
		
		matching_problem.Info = info
		if (matching_problem.it % 10 == 0):# and (c.data.cpu().numpy()[0] < matching_problem.bestc):
			matching_problem.bestc = c.data.cpu().numpy()[0]
			q1,p1,g1 = _HamiltonianCarrying(q0, p0, g0, params["deformation"], q_mu = Q1_mu)
			
			q1 = q1.data.cpu().numpy()
			p1 = p1.data.cpu().numpy()
			g1 = g1.data.cpu().numpy()
			
			Q1 = Curve(q1, connec_plot) ; G1 = Curve(g1, cgrid )
			DisplayShoot( Q0, G0,       p0.data.cpu().numpy(), 
			              Q1, G1, Xt, info.data.cpu().numpy(),
			              matching_problem.it, scale_momentum, scale_attach,
			              attach_type = params["data_attachment"]["formula"])
			              
			if show_trajectories :
				Qts, Pts, Gts = _HamiltonianCarrying(q0, p0, g0, params["deformation"], q_mu = Q1_mu,
				                                     trajectory = True, endtime = 2.)
				
				# N.B.: if landmarks,  connec = None
				Qts = [Curve( Qt.data.cpu().numpy(), connec_plot ) for Qt in Qts] 
				Pts = [Pt.data.cpu().numpy() for Pt in Pts]
				Gts = [Curve( Gt.data.cpu().numpy(), cgrid  ) for Gt in Gts]
				DisplayTrajectory(Qts, Pts[0], scale_momentum * 2, Xt, matching_problem.it)
		
		print('Iteration : ', matching_problem.it, ', cost : ', c.data.cpu().numpy(), 
		                                            ' info : ', info.data.cpu().numpy().shape)
		matching_problem.it += 1
		return c
	matching_problem.bestc = np.inf ; matching_problem.it = 0 ; matching_problem.Info = None
	
	if False : # We cannot rely on PyTorch's LBFGS just yet, as they did not implement line search...
		# The PyTorch factorization of minimisation problems is a bit unusual...
		optimizer = torch.optim.LBFGS(
						[p0],
						lr               = 1.,      # Learning rate
						max_iter         = 1000, 
						tolerance_change = .000001, 
						history_size     = 10)
		#optimizer = torch.optim.Adam(
		#				[p0])
		time1 = time.time()
		def closure():
			"Encapsulates the matching problem + display."
			optimizer.zero_grad()
			c = matching_problem(p0)
			c.backward()
			return c
		
		for it in range(100) :
			optimizer.step(closure)
	else : # So, let's fall back on the good old rock-solid Fortran routines...
		from scipy.optimize import minimize
		def numpy_matching_problem(num_p0) :
			"""
			Wraps matching_problem into a 'float64'-vector routine,
			as expected by scipy.optimize.
			"""
			num_p0 = params["learning_rate"] * num_p0.astype('float64')
			tor_p0 = Variable(torch.from_numpy(num_p0.reshape(Q0_points.shape)).type(dtype), 
			                  requires_grad=True ).contiguous()
			c = matching_problem(tor_p0)
			#print('Cost : ', c)
			dp_c = torch.autograd.grad( c, [tor_p0] )[0]
			dp_c = params["learning_rate"] * dp_c.data.numpy()
			# The fortran routines used by scipy.optimize expect float64 vectors
			# instead of the gpu-friendly float32 matrices
			return (c.data.numpy(), dp_c.ravel().astype('float64'))
			
		time1 = time.time()
		res = minimize( numpy_matching_problem, # function to minimize
						(0.*Q0_points).ravel(), # starting estimate
						method = 'L-BFGS-B',  # an order 2 method
						jac = True,           # matching_problems also returns the gradient
						options = dict(
							maxiter = 1000,
							ftol    = .00001, # Don't bother fitting the shapes to float precision
							maxcor  = 10      # Number of previous gradients used to approximate the Hessian
						))
	time2 = time.time()
	
	time2 = time.time()
	return p0, matching_problem.Info

if __name__ == '__main__' :
	import matplotlib.pyplot as plt
	plt.ion()
	plt.show()
	
	deformation_example = 1 ; attachment_example  = 1 ;
	dataset = "ameoba"      ; npoints             = 1000
	
	# A few deformation kernels =================================================================
	if   deformation_example == 1 : # Standard Gaussian kernel
		params_def = dict(
			mode      = "gaussian",
			radius    = [.15],
			normalize = False
		)
	elif deformation_example == 2 :  # Normalized Gaussian   -> plate tectonics
		params_def = dict(
			mode      = "gaussian",
			radius    = [.15],
			normalize = True
		)
	elif deformation_example == 3 :  # Normalized Heavy tail -> translation awareness
		params_def = dict(
			mode      = "energy",
			radius    = [.05],
			normalize = True
		)
	# A few data attachment terms ===============================================================
	if attachment_example  == 1 :   # Standard (heavy-tail) kernel attachment
		params_att = dict(
			formula   = "kernel",
			mode      = "energy",
			radius    = [.05]
		)
	elif attachment_example == 2 :  # Optimal Transport. N.B.: I didn't really implement it all yet...
		params_att = dict(
			formula   = "OT",
			epsilon   = (.1)**2,   # regularization parameter
			rho       = (.5) **2,   # unbalanced transport (See PhD Th. of Lenaic Chizat)
			niter     = 10000,      # max niter in the sinkhorn loop
			tau       = -.8,        # nesterov-like acceleration
			radius    = [0.]
		)
	elif attachment_example == 3 :  # L2
		params_att = dict(
			formula   = "L2",
			radius    = [0.]
		)
		
	# A few datasets ============================================================================
	if dataset == "ameoba" :
		params = dict(
			data_type        = "curves",
			transport_action = "image",
			deformation      = params_def,
			data_attachment  = params_att
		)
		Q0 = Curve.from_file('amoeba_1.png', npoints = npoints) # Load source...
		Xt = Curve.from_file('amoeba_2.png', npoints = npoints) # and target.
	
	elif dataset == "skulls" :
		params = dict(
			data_type        = "curves",
			transport_action = "image",
			deformation      = params_def,
			data_attachment  = params_att
		)
		Q0 = Curve.from_file('australopithecus.vtk') # Load source...
		Xt = Curve.from_file('sapiens.vtk')          # and target.
	
	if dataset == "ameoba_measures" :
		params = dict(
			data_type        = "landmarks",
			transport_action = "measure",
			deformation      = params_def,
			data_attachment  = params_att
		)
		Q0 = Curve.from_file('amoeba_1.png', npoints = npoints) # Load source...
		Xt = Curve.from_file('amoeba_2.png', npoints = npoints) # and target.
		Q0 = Q0.to_measure()
		Xt = Xt.to_measure()
		
	elif dataset == "mario" :
		params = dict(
			data_type        = "landmarks",
			transport_action = "measure",
			deformation      = params_def,
			data_attachment  = params_att
		)
		Q0_x = np.array( [[.2,.7],[.2,.3]]) # Source...
		Xt_x = np.array( [[.8,.71],[.8,.3]]) # and target.
		
		Q0_mu = np.ones((len(Q0_x),))
		Xt_mu = np.ones((len(Xt_x),))
		Q0 = [np.array(Q0_x).astype('float32'), np.array(Q0_mu).astype('float32')]
		Xt = [np.array(Xt_x).astype('float32'), np.array(Xt_mu).astype('float32')]
		
	elif dataset == "GFSW03" :
		params = dict(
			data_type        = "landmarks",
			transport_action = "measure",
			deformation      = params_def,
			data_attachment  = params_att
		)
		t = np.linspace(0, 2*np.pi, npoints // 2 + 1)[:-1]

		# Shapes to be matched, given by their Fourier series :
		Q0_1 = np.vstack( (     np.cos(t) + .2 * np.sin(4*t) + .2 * np.cos(7*t) + .05 * np.cos(13*t),
						    1 * np.sin(t) + .3 * np.cos(4*t) + .2 * np.sin(7*t) + .05 * np.sin(13*t)  ) 
					  ).T
		Xt_1 = np.vstack( (np.cos(t) + .2 * np.sin(4*t) + .05 * np.cos(13*t),
						   1 * np.sin(t) + .3 * np.cos(4*t) + .05 * np.sin(13*t)  ) 
					  ).T
		
		Q0_2 = np.vstack( (1.5*np.cos(t),     np.sin(t)) ).T
		Xt_2 = np.vstack( (    np.cos(t), 1.5*np.sin(t)) ).T
		
		Q0_x = np.vstack(( .1* Q0_1 + np.array([.25,.75]),  .1*Q0_2 + np.array([.75,.5])))
		Xt_x = np.vstack(( .1* Xt_1 + np.array([.75,.50]),  .1*Xt_2 + np.array([.25,.25])))
		
		Q01_mu = 10*np.ones((len(Q0_1),))
		Xt1_mu = 10*np.ones((len(Xt_1),))
		Q02_mu =    np.ones((len(Q0_2),))
		Xt2_mu =    np.ones((len(Xt_2),))
		Q0_mu = np.hstack((Q01_mu,Q02_mu))
		Xt_mu = np.hstack((Xt1_mu,Xt2_mu))
		
		# normalize so that parameters make sense whatever the number of points
		Q0_mu = Q0_mu / np.sum(Q0_mu)
		Xt_mu = Xt_mu / np.sum(Xt_mu)
		
		Q0 = [np.array(Q0_x).astype('float32'), np.array(Q0_mu).astype('float32')]
		Xt = [np.array(Xt_x).astype('float32'), np.array(Xt_mu).astype('float32')]

	
	# Run the test (+rescale the quivers as needed...) ==========================================
	params["learning_rate"] = .001  # L-BFGS will adjust the scale after a few iterations, but you'd
	                                # better start slowly if you don't want to get lost at step=1...
	p0, info = perform_matching( Q0, Xt, params, scale_momentum = 2., scale_attach = 1.) 


