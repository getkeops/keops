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

if False :
	Source = Curve.from_file(FOLDER+"data/amoeba_1.png", npoints=200)
	Target = Curve.from_file(FOLDER+"data/amoeba_2.png", npoints=200)
else :
	Source = Surface.from_file(FOLDER+"data/venus_1.vtk")
	Target = Surface.from_file(FOLDER+"data/venus_4.vtk")
	#Target.points.data[:,2] += 1. # Let's shift the target a little bit...

def scal_to_var(x) :
	return Variable(Tensor([x])).type(dtype)

s_def = .1
s_att = .1
eps   = scal_to_var(s_att**2)
backend = "auto"


G  = 1/eps
H  = scal_to_var(.1)
I  = scal_to_var(.1)

features = "locations+directions"

# Create a custom kernel, purely in log-domain, for the Wasserstein/Sinkhorn cost.
# formula_log = libkp backend, routine_log = pytorch backend : a good "safety check" against typo errors !
if   features == "locations":
	kernel              = Kernel("gaussian(x,y)")
	params_kernel       = G

elif features == "locations+directions" :
	kernel              = Kernel()
	kernel.features     = "locations+directions"
	kernel.formula_log  = "( -Cst(G)*SqDist(X,Y) * (IntCst(1) + Cst(H)*(IntCst(1)-Pow((U,V),2) ) + Cst(I)*SqDist(S,T) ) )"
	kernel.routine_log  = lambda g=None, xmy2=None, h=None, usv=None, i=None, smt2=None, **kwargs :\
										-g*xmy2        * (    1     +      h*(1-usv**2)           +      i*smt2        )
	params_kernel       = (G,H)

elif features == "locations+directions+values" :
	kernel             = Kernel()
	kernel.features    = "locations+directions+values"
	kernel.formula_log = "( -Cst(G)*SqDist(X,Y) * (IntCst(1) + Cst(H)*(IntCst(1)-Pow((U,V),2) ) + Cst(I)*SqDist(S,T) ) )"
	kernel.routine_log = lambda g=None, xmy2=None, h=None, usv=None, i=None, smt2=None, **kwargs :\
										-g*xmy2        * (    1     +      h*(1-usv**2)           +      i*smt2        )
	params_kernel       = (G,H,I)

params = {
	"weight_regularization" : .1,               # MANDATORY
	"weight_data_attachment": 1.,               # MANDATORY

	"deformation_model" : {
		"id"         : Kernel("gaussian(x,y)"),        # MANDATORY
		"gamma"      : scal_to_var(1/s_def**2),      # MANDATORY
		"backend"    : backend,                 # optional  (["auto"], "pytorch", "CPU", "GPU_1D", "GPU_2D")
		"normalize"  : False,           # optional  ([False], True)
	},

	"data_attachment"   : {
		"formula"            : "wasserstein",
		"features"           : kernel.features,
		"kernel" : {"id"     : kernel ,
					"gamma"  : params_kernel ,
					"backend": backend                 },
		"epsilon"            : eps,
		"rho"                : -1,
		"tau"                : -.8,
		"nits"               : 20,
		"tol"                : 1e-7,
		"transport_plan"     : "none",
	},

	"optimization" : {                          # optional
		"method"             : "L-BFGS",        # optional
		"nits"               : 100,             # optional
		"nlogs"              : 1,              # optional
		"tol"                : 1e-7,            # optional

		"lr"                 : .01,            # optional
		"maxcor"             : 10,              # optional (L-BFGS)
	},

	#"display" : {
	#	"limits"             : [0,1,0,1],
	#	"grid_ticks"         : ((0,1,11),)*2,
	#	"template"           : False,
	#},
	"save" : {                                  # MANDATORY
		"output_directory"   : FOLDER+"output/sinkhorn/",# MANDATORY
	}
}

# Define our (simplistic) matching model
Model = GeodesicMatching(Source)

# Train it
FitModel(params, Model, Target)

# That's it :-)
