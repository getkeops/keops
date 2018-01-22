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

from .toolbox               import shapes
from .toolbox.shapes        import Curve
from .toolbox.matching      import GeodesicMatching
from .toolbox.model_fitting import FitModel

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

s_def = .015
s_att = .05
backend = "auto"
def scal_to_var(x) :
	return Variable(Tensor([x])).type(dtype)

params = {
	"weight_regularization" : .1,               # MANDATORY
	"weight_data_attachment": 1.,               # MANDATORY

	"deformation_model" : {
		"name"  : "energy",                     # MANDATORY
		"gamma" : scal_to_var(1/s_def**2),      # MANDATORY
		"backend"    : backend,                 # optional  (["auto"], "pytorch", "CPU", "GPU_1D", "GPU_2D")
		"normalize"          : False,           # optional  ([False], True)
	},

	"data_attachment"   : {
		"formula"            : "kernel",        # MANDATORY ("L2", "kernel", "wasserstein", "sinkhorn")
		"features"           : "locations",     # optional  (["locations"], "locations+normals")

		# Kernel-specific parameters:
		"name"       : "energy",                # MANDATORY (if "formula"=="kernel")
		"gamma"      : scal_to_var(1/s_att**2), # MANDATORY (if "formula"=="kernel")
		"backend"    : backend,                 # optional  (["auto"], "pytorch", "CPU", "GPU_1D", "GPU_2D")
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
		"method"             : "L-BFGS",        # optional
		"nits"               : 100,             # optional
		"nlogs"              : 10,              # optional
		"tol"                : 1e-7,            # optional

		"lr"                 : .001,            # optional
		"eps"                : .01,             # optional (Adam)
		"maxcor"             : 10,              # optional (L-BFGS)
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
