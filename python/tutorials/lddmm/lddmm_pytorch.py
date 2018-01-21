# LDDMM registration using PyTorch
# Author : Jean Feydy

# Note about function names :
#  - MyFunction  is an input-output code
#  - my_routine  is a numpy routine
#  - _my_formula is a PyTorch symbolic function

# Import the relevant tools
import time                 # to measure performance
import numpy as np          # standard array library
import torch
from   torch          import Tensor
from   torch.autograd import Variable

import os
FOLDER = os.path.dirname(os.path.abspath(__file__))+os.path.sep

from   .    import shapes
from .shapes import Curve
from .matching import GeodesicMatching
from .model_fitting import FitModel

import matplotlib.pyplot as plt
plt.ion()
plt.show()


# Choose the storage place for our data : CPU (host) or GPU (device) memory.
use_cuda = torch.cuda.is_available()
dtype    = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
dtypeint = torch.cuda.LongTensor  if use_cuda else torch.LongTensor

# Make sure that everybody's on the same wavelength:
shapes.dtype = dtype ; shapes.dtypeint = dtypeint

Source = Curve.from_file(FOLDER+"data/amoeba_1.png", npoints=1000)
Target = Curve.from_file(FOLDER+"data/amoeba_2.png", npoints=1000)

s_def = .2
s_att = .25
backend = "GPU_1D"
def scal_to_var(x) :
	return Variable(Tensor([x])).type(dtype)

params = {
	"weight_regularization" : .1,               # MANDATORY
	"weight_data_attachment": 1.,               # MANDATORY

	"deformation_model" : {
		"name"  : "gaussian",                   # MANDATORY
		"gamma" : scal_to_var(1/s_def**2),      # MANDATORY
		"backend"    : backend,                 # optional  (["auto"], "pytorch", "CPU", "GPU_1D", "GPU_2D")
		"normalize"          : False,           # optional  ([False], True)
	},

	"data_attachment"   : {
		"formula"            : "kernel",        # MANDATORY ("L2", "kernel", "wasserstein", "sinkhorn")
		"features"           : "locations",     # optional  (["locations"], "locations+normals")

		# Kernel-specific parameters:
		"name"       : "gaussian",              # MANDATORY (if "formula"=="kernel")
		"gamma"      : scal_to_var(1/s_att**2), # MANDATORY (if "formula"=="kernel")
		"backend"    : backend,                  # optional  (["auto"], "pytorch", "CPU", "GPU_1D", "GPU_2D")
		"kernel_heatmap_range" : (-2,2,100),    # optional

		# Wasserstein+Sinkhorn -specific parameters:
		"epsilon"       : scal_to_var(1/.5**2), # MANDATORY (if "formula"=="wasserstein" or "sinkhorn")
		"kernel"        : {                     # optional
			"name"  : "gaussian" ,              #     ...
			"gamma" : 1/scal_to_var(1/.5**2),   #     ...
			"backend": backend },               #     ...
		"rho"                : -1,              # optional
		"tau"                : 0.,              # optional
		"nits"               : 1000,            # optional
		"tol"                : 1e-5,            # optional
	},

	"optimization" : {                          # optional
		"method"             : "Adam",          # optional
		"nits"               : 100,             # optional
		"nlogs"              : 10,              # optional
		"tol"                : 1e-7,            # optional

		"lr"                 : .001,            # optional
		"eps"                : .01,             # optional
	},

	"display" : {                               # optional
		"limits"             : [0,1,0,1],       # MANDATORY
		"grid"               : True,            # optional
		"grid_ticks"         : ((0,1,11),)*2,   # optional (default : ((-1,1,11),)*dim)
		"grid_color"         : (.8,.8,.8),      # optional
		"grid_linewidth"     : 1,               # optional

		"template"           : False,           # optional
		"template_color"     : "b",             # optional
		"template_linewidth" : 2,               # optional

		"target"             : True,            # optional
		"target_color"       : (.76, .29, 1.),  # optional
		"target_linewidth"   : 2,               # optional

		"model"              : True,            # optional
		"model_color"        : "rainbow",       # optional
		"model_linewidth"    : 2,               # optional
	},

	"save" : {                                  # MANDATORY
		"output_directory"   : FOLDER+"output/",# MANDATORY
		"template"           : True,            # optional
		"model"              : True,            # optional
		"target"             : True,            # optional
		"momentum"           : True,            # optional
		"info"               : True,            # optional
		"movie"              : False,           # optional
	}
}

# Define our (simplistic) matching model
Model = GeodesicMatching(Source)

# Train it
FitModel(params, Model, Target)

# That's it :-)

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
