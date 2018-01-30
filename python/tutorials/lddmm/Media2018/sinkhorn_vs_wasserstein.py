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

from ..toolbox                import shapes
from ..toolbox.shapes         import Curve, Surface
from ..toolbox.matching       import GeodesicMatching
from ..toolbox.model_fitting  import FitModel
from ..toolbox.kernel_product import Kernel


import matplotlib.pyplot as plt
plt.ion()
plt.show()

# Choose the storage place for our data : CPU (host) or GPU (device) memory.
use_cuda = torch.cuda.is_available()
dtype    = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
dtypeint = torch.cuda.LongTensor  if use_cuda else torch.LongTensor

# Make sure that everybody's on the same wavelength:
shapes.dtype = dtype ; shapes.dtypeint = dtypeint

Source = Curve.from_file(FOLDER+"data/amoeba_1.png", npoints=500)
Target = Curve.from_file(FOLDER+"data/amoeba_2.png", npoints=500)

def scal_to_var(x) :
	return Variable(Tensor([x])).type(dtype)

s_def = .1
s_att = .1
eps   = scal_to_var(s_att**2)
backend = "auto"


G  = 1/eps           # "gamma" of the gaussian
H  = scal_to_var(2.) # weight in front of (u,v)    (orientations)

features = "locations+directions"

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


params = {
	"weight_regularization" : .01,
	"weight_data_attachment": 1.,

	"deformation_model" : {
		"id"         : Kernel("gaussian(x,y)"),
		"gamma"      : scal_to_var(1/s_def**2),
		"backend"    : backend,                  # optional  (["auto"], "pytorch", "CPU", "GPU_1D", "GPU_2D")
		"normalize"  : False,                    # optional  ([False], True)
	},

	"data_attachment"   : {
		"formula"            : "wasserstein", # Choose sinkhorn, then wasserstein

		# Parameters for OT:
		"cost"               : "dual",
		"features"           : kernel.features,
		"kernel" : {"id"     : kernel ,
					"gamma"  : params_kernel ,
					"backend": backend                 },
		"epsilon"            : eps,
		"rho"                : 1.,   # < 0 -> no unbalanced transport
		"tau"                : 0.,   # Using acceleration with nits < ~40 leads to *very* unstable transport plans
		"nits"               : 50,
		"tol"                : 1e-7,
		"transport_plan"     : "minimal_symmetric",
	},
	"optimization" : {
		"nlogs"              : 1,
		"tol"                : 1e-6,
	},
	"display" : {
		"limits"             : [0,1,0,1],
		"grid"               : False,
		"grid_ticks"         : ((0,1,21),(0,1,21)),
		"template"           : False,
		"target_color"       : (0.,0.,.8, .5),
		"model_color"        : (.8,0.,0.),
		"model_linewidth"    : 3,
		"info_color"         : (.8, .9, 1.,1.),
		"info_color_a"       : (.8, .4, .4, .2),
		"info_color_b"       : (.4, .4, .8, .2),
		"show_axis"          : False,
		"model_gradient"     : False,
	},
	"save" : {                                  # MANDATORY
		"output_directory"   : FOLDER+"output/sinkhorn_vs_wasserstein/wasserstein_01/",# MANDATORY
	}
}


for (scale_att, suffix) in [ (0.1, "01"), (0.05, "005"), (0.01,"001"), (0.5, "05")] :	
	for mode in ["sinkhorn", "wasserstein"] :
		params["save"]["output_directory"] = FOLDER+"output/sinkhorn_vs_wasserstein_long/"+mode+"_"+suffix+"/"
		params["data_attachment"]["formula"] = mode
		eps   = scal_to_var(scale_att**2)
		G  = 1/eps
		params["data_attachment"]["epsilon"] = eps
		params["data_attachment"]["kernel"]["gamma"] = (G,H)
		Model = GeodesicMatching(Source)
		FitModel(params, Model, Target)

# That's it :-)
