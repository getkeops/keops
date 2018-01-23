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

Source = Curve.from_file(FOLDER+"data/amoeba_1.png", npoints=300)
Target = Curve.from_file(FOLDER+"data/amoeba_2.png", npoints=300)

s_def = .015
s_att = .05
backend = "auto"
def scal_to_var(x) :
	return Variable(Tensor([x])).type(dtype)

params = {
	"weight_regularization" : .1,               # MANDATORY
	"weight_data_attachment": 1.,               # MANDATORY

	"deformation_model" : {
		"name"  : Kernel("energy(x,y)"),        # MANDATORY
		"gamma" : scal_to_var(1/s_def**2),      # MANDATORY
		"backend"    : backend,                 # optional  (["auto"], "pytorch", "CPU", "GPU_1D", "GPU_2D")
		"normalize"          : False,           # optional  ([False], True)
	},

	"data_attachment"   : {
		"formula"            : "wasserstein",   # MANDATORY
		"features"           : "locations",     # MANDATORY  (["locations"], "locations+normals")

		"epsilon"            : scal_to_var(1/s_att**2),
		"rho"                : 10,
		"tau"                : -.8,
		"nits"               : 300,
		"tol"                : 1e-5,
	},

	"optimization" : {                          # optional
		"method"             : "L-BFGS",        # optional
		"nits"               : 100,             # optional
		"nlogs"              : 1,              # optional
		"tol"                : 1e-7,            # optional

		"lr"                 : .001,             # optional
		"maxcor"             : 10,              # optional (L-BFGS)
	},

	"display" : {
		"limits"             : [0,1,0,1],
		"grid_ticks"         : ((0,1,11),)*2,
		"template"           : False,
	},
	"save" : {                                  # MANDATORY
		"output_directory"   : FOLDER+"output/sinkhorn/",# MANDATORY
	}
}

# Define our (simplistic) matching model
Model = GeodesicMatching(Source)

# Train it
FitModel(params, Model, Target)

# That's it :-)
