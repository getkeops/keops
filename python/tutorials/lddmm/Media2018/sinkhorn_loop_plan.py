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

Source = Curve.from_file(FOLDER+"data/amoeba_1.png", npoints=100)
Target = Curve.from_file(FOLDER+"data/amoeba_2.png", npoints=100)

def scal_to_var(x) :
	return Variable(Tensor([x])).type(dtype)

s_def = .1
s_att = .1
eps   = scal_to_var(s_att**2)
backend = "pytorch"


G  = 1/eps           # "gamma" of the gaussian

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
		"id"     : kernel ,
		"gamma"  : params_kernel ,
		"backend": backend,
		"kernel_heatmap_range" : (0,1,100),

		# Parameters for OT:
		"cost"               : "dual",
		"features"           : features,
		"normalize"          : False,
		"kernel" : {"id"     : kernel ,
					"gamma"  : params_kernel ,
					"backend": backend                 },
		"epsilon"            : eps,
		"rho"                : 1.,            # < 0 -> no unbalanced transport
		"tau"                : 0.,              # Using acceleration with nits < ~40 leads to *very* unstable transport plans
		"nits"               : 20,
		"tol"                : 1e-10,
		"transport_plan"     : "minimal_symmetric",
		"frac_mass_per_line" : 0.01,
	},
	"optimization" : {
		"nits"               : 1, # We're not here to bother with LDDMM matching...
		"nlogs"              : 1,
	},
	"display" : {
		"limits"             : [0,1,0,1],
		"grid"               : False,
		"template"           : False,
		"target_color"       :   (0.,0.,.8),
		"model_color"        :   (.8,0.,0.),
		"info_color"         : (.8, .9, 1.,.5),
		"info_linewidth"     : 2.,
		"show_axis"          : False,
	},
	"save" : {                                  # MANDATORY
		"output_directory"   : FOLDER+"output/sinkhorn_plan/",# MANDATORY
	}
}

fig_model = plt.figure(figsize=(10,10), dpi=100)
ax_model  = plt.subplot(1,1,1)
ax_model.autoscale(tight=True)
for (normalize, rho, scale) in [(True, -1, 1), (False, 1., 2)] :
	params["data_attachment"]["normalize"]  = normalize
	params["data_attachment"]["rho"]        = rho

	for formula in ["sinkhorn", "wasserstein"] :
		params["data_attachment"]["formula"]  = formula
		for mode in ["dual", "primal", "spring"] :
			params["data_attachment"]["cost"] =mode
			for i in [0,1,2,3,4,5,6,7,8,9,10,15,20,49,50,100,1000] :
				for half_step, suffix in [ (False, "a"), (True, "b")] :
					params["data_attachment"]["nits"]           = i
					params["display"]["model_gradient_scale"]   = scale
					params["data_attachment"]["end_on_target"]  = half_step

					Model = GeodesicMatching(Source)
					cost,info,model = Model.cost(params, Target, info=True )
					cost.backward()

					ax_model.clear()
					Model.plot(ax_model, params, target=Target, info=info)
					ax_model.axis(params["display"]["limits"]) ; ax_model.set_aspect('equal') ; 
					if not params["display"].get("show_axis", True) : ax_model.axis('off')
					plt.draw() ; plt.pause(0.01)

					norm_suff = "_normalized" if normalize else ""
					screenshot_filename = FOLDER+"output/sinkhorn_plan/" \
										+formula+"_"+mode+norm_suff+"/nits_"+str(i)+suffix+'.png'
					os.makedirs(os.path.dirname(screenshot_filename), exist_ok=True)
					fig_model.savefig( screenshot_filename, bbox_inches='tight' )

# That's it :-)
