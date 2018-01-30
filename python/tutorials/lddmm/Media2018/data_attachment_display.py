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
use_cuda = False # torch.cuda.is_available()
dtype    = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
dtypeint = torch.cuda.LongTensor  if use_cuda else torch.LongTensor

# Make sure that everybody's on the same wavelength:
shapes.dtype = dtype ; shapes.dtypeint = dtypeint

if False :
	Source = Curve.from_file(FOLDER+"data/shape_left_2.png", npoints=15)
	Target = Curve.from_file(FOLDER+"data/shape_left_2.png", npoints=15)
	Target.points.data += torch.Tensor([.6,0.]).type(dtype)
else :
	Source = Curve.from_file(FOLDER+"data/amoeba_1.png", npoints=200)
	Target = Curve.from_file(FOLDER+"data/amoeba_2.png", npoints=200)


def scal_to_var(x) :
	return Variable(Tensor([x])).type(dtype)

s_def = .1
s_att = .01
eps   = scal_to_var(s_att**2)
backend = "pytorch"


G  = 1/eps           # "gamma" of the gaussian
H  = scal_to_var(5.) # weight in front of (u,v)    (orientations)
I  = scal_to_var(5.) # weight in front of |s-t|^2  (signals)

features = "locations"
kernel              = Kernel("gaussian(x,y)")
params_kernel       = G


params = {
	"weight_regularization" : .00001,
	"weight_data_attachment": 1.,

	"deformation_model" : {
		"id"         : Kernel("gaussian(x,y)"),
		"gamma"      : scal_to_var(1/s_def**2),
		"backend"    : backend,                  # optional  (["auto"], "pytorch", "CPU", "GPU_1D", "GPU_2D")
		"normalize"  : False,                    # optional  ([False], True)
	},

	"data_attachment"   : {
		"formula"            : "wasserstein",

		# Just in case you intend to use a kernel fidelity:
		"id"     : Kernel("energy(x,y)") ,
		"gamma"  : scal_to_var(1/.01**2) ,
		"backend": backend,
		"kernel_heatmap_range" : (-0.3,1.3,300),

		# Parameters for OT:
		"cost"               : "dual",
		"features"           : kernel.features,
		"kernel" : {"id"     : kernel ,
					"gamma"  : params_kernel ,
					"backend": backend                 },
		"epsilon"            : eps,
		"rho"                : -1.,            # < 0 -> no unbalanced transport
		"tau"                : 0.,              # Using acceleration with nits < ~40 leads to *very* unstable transport plans
		"nits"               : 100,
		"tol"                : 1e-7,
		"transport_plan"     : "full",
	},
	"optimization" : {
		"method"             : "Adam",
		"nlogs"              : 1,
		"nits"               : 1,
	},
	"display" : {
		"limits"             : [-.3,1.3,-.3,1.3],
		"grid"               : False,
		"template"           : False,
		"target_color"       :   (0.,0.,.8),
		"model_color"        :   (.8,0.,0.),
		"info_color"         : (.8, .9, 1.,.5),
		"target_linewidth"   : 4,
		"model_linewidth"    : 4,
		"info_linewidth"     : 4.,
		"show_axis"          : False,
		"model_gradient"     : False, # Renormalize stuff with a "negative" scale
	},
	"save" : {                                  # MANDATORY
		"output_directory"   : FOLDER+"output/data_attachment_gradients/OT/",# MANDATORY
	}
}

if True :
	settings = [("gaussian(x,y)", .01, "_empty", 1000),
				("energy(x,y)",   .01,  "_heavy", .25),
				("gaussian(x,y)", .02, "_small", None), 
				("gaussian(x,y)", .2,  "_large", None)]

for formula in ["kernel", "likelihood_symmetric"] :
	for (name, scale, suffix, maxval) in settings :
		params["data_attachment"]["formula"] = formula
		params["data_attachment"]["id"]      = Kernel(name)
		params["data_attachment"]["gamma"]   = scal_to_var(1/scale**2)
		params["display"]["kernel_heatmap_max"] = maxval

		params["save"]["output_directory"] = FOLDER+"output/data_attachment_display/"+formula+suffix+"/"
		Model = GeodesicMatching(Source)
		FitModel(params, Model, Target)
