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

from .toolbox                import shapes
from .toolbox.shapes         import Curve, Surface
from .toolbox.matching       import GeodesicMatching
from .toolbox.model_fitting  import FitModel
from .toolbox.kernel_product import Kernel


import matplotlib.pyplot as plt
plt.ion()
plt.show()

# Choose the storage place for our data : CPU (host) or GPU (device) memory.
use_cuda = torch.cuda.is_available()
dtype    = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
dtypeint = torch.cuda.LongTensor  if use_cuda else torch.LongTensor

# Make sure that everybody's on the same wavelength:
shapes.dtype = dtype ; shapes.dtypeint = dtypeint

if False : # (Jean:) I don't have any fshape curve at hand to test, so this may be a bit buggy...
	Source = Curve.from_file(FOLDER+"data/amoeba_1.png", npoints=200)
	Target = Curve.from_file(FOLDER+"data/amoeba_2.png", npoints=200)
else :
	Source = Surface.from_file(FOLDER+"data/venus_1.vtk")
	Target = Surface.from_file(FOLDER+"data/venus_4.vtk")
	Target.points.data[:,2] += 1.5 # Let's shift the target a little bit...

def scal_to_var(x) :
	return Variable(Tensor([x])).type(dtype)

s_def = .1
s_att = .1
eps   = scal_to_var(s_att**2)
backend = "auto"


G  = 1/eps           # "gamma" of the gaussian
H  = scal_to_var(5.) # weight in front of (u,v)    (orientations)
I  = scal_to_var(5.) # weight in front of |s-t|^2  (signals)

features = "locations"

# Create a custom kernel, purely in log-domain, for the Wasserstein/Sinkhorn cost.
# formula_log = libkp backend, routine_log = pytorch backend : a good "safety check" against typo errors !
# N.B.: the shorter the formula, the faster the kernel...
if   features == "locations": # Fast as hell kernel, to compute a regular Wasserstein distance (the "mathematical" one)
	kernel              = Kernel("gaussian(x,y)")
	params_kernel       = G

elif features == "locations+directions" : # Generalization to varifolds - Cf eq.(11) of the MICCAI paper
	kernel              = Kernel()
	kernel.features     = "locations+directions"
	kernel.formula_log  = "( -Cst(G)*SqDist(X,Y) * (IntCst(1) + Cst(H)*(IntCst(1)-Pow((U,V),2) ) ) )"
	kernel.routine_log  = lambda g=None, xmy2=None, h=None, usv=None, i=None, smt2=None, **kwargs :\
								  -g*xmy2        * (    1     +      h*(1-usv**2)   )
	params_kernel       = (G,H)

elif features == "locations+directions+values" : # Generalization to functional varifolds
	kernel             = Kernel()
	kernel.features    = "locations+directions+values"
	kernel.formula_log = "( -Cst(G)*SqDist(X,Y) * (IntCst(1) + Cst(H)*(IntCst(1)-Pow((U,V),2) ) + Cst(I)*SqDist(S,T) ) )"
	kernel.routine_log = lambda g=None, xmy2=None, h=None, usv=None, i=None, smt2=None, **kwargs :\
								 -g*xmy2        * (    1     +      h*(1-usv**2)                +      i*smt2        )
	params_kernel       = (G,H,I)

params = {
	"weight_regularization" : .1,
	"weight_data_attachment": 1.,

	"deformation_model" : {
		"id"         : Kernel("energy(x,y)"),
		"gamma"      : scal_to_var(1/s_def**2),
		"backend"    : backend,                  # optional  (["auto"], "pytorch", "CPU", "GPU_1D", "GPU_2D")
		"normalize"  : False,                    # optional  ([False], True)
	},

	"data_attachment"   : {
		#"formula"            : "kernel",
		"formula"            : "wasserstein",
		#"formula"            : "sinkhorn",

		# Just in case you intend to use a kernel fidelity:
		"id"     : kernel ,
		"gamma"  : params_kernel ,
		"backend": backend,

		# Parameters for OT:
		"cost"               : "primal",
		"features"           : kernel.features,
		"kernel" : {"id"     : kernel ,
					"gamma"  : params_kernel ,
					"backend": backend                 },
		"epsilon"            : eps,
		"rho"                : -1,              # < 0 -> no unbalanced transport
		"tau"                : 0.,              # Using acceleration with nits < ~40 leads to *very* unstable transport plans
		"nits"               : 20,
		"tol"                : 1e-7,
		"transport_plan"     : "minimal",       # Visualization parameter:
												# "none" is default
												# "full" is ok for small point clouds
		                                        # "minimal" plots (x -> (Gamma*y)/mu), i.e. "where the source wants to go"
												#     Since we were careful enough to "always finish with a projecion
												#     on the mu-constraint" to get a good gradient wrt. x, 
												#     this always looks good... 
												# "minimal_symmetric" plots "minimal" both (x -> (Gamma*y)/mu)
												#     and (y -> (Gamma*x)/nu). It is thus an honest representation
												#     of "the non-convergence" of the Sinkhorn algorithm.
	},

	"optimization" : {                          # optional
		"method"             : "L-BFGS",        # optional
		"nits"               : 1000,            # optional
		"nlogs"              : 1,               # optional
		"tol"                : 1e-7,            # optional

		"lr"                 : .001,            # optional
		"maxcor"             : 10,              # optional (L-BFGS)
	},

	#"display" : {
	#	"limits"             : [0,1,0,1],
	#	"grid_ticks"         : ((0,1,11),)*2,
	#	"template"           : False,
	#},
	"save" : {                                  # MANDATORY
		"output_directory"   : FOLDER+"output/wasserstein_measure/",# MANDATORY
	}
}

# Define our (simplistic) matching model
Model = GeodesicMatching(Source)

# Train it
FitModel(params, Model, Target)

# That's it :-)
