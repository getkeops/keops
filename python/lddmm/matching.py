# Main demo script. (see the __main__ section of the code)
import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '..')

# Import the relevant tools
import time                 # to measure performance
import numpy as np          # standard array library
import torch
from   torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt

from numpy.random import normal

# No need for a ~/.theanorc file anymore !
use_cuda = False #torch.cuda.is_available()
dtype    = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
dtypeint = torch.cuda.LongTensor  if use_cuda else torch.LongTensor

from input_output    import GridData, DisplayShoot, DisplayTrajectory
from shooting        import _Hqp, _HamiltonianShooting, _HamiltonianCarrying, _HamiltonianTrajectory
from data_attachment import _data_attachment
from curve           import Curve

# Cost function and derivatives =================================================================

def _cost( q,p, xt_measure, connec, radius_def, radius_att, attach_mode = "OT", q1_mu = None ) :
	"""
	Returns a total cost, sum of a small regularization term and the data attachment.
	.. math ::
	
		C(q_0, p_0) = .01 * H(q0,p0) + 1 * A(q_1, x_t)
	
	Needless to say, the weights can be tuned according to the signal-to-noise ratio.
	"""
	q1 = _HamiltonianShooting(q,p,radius_def, q_mu = q1_mu)[0]  # Geodesic shooting from q0 to q1
	# To compute a data attachment cost, we need the set of vertices 'q1' into a measure.
	
	if q1_mu is None :
		q1_measure  = Curve._vertices_to_measure( q1, connec ) 
	else :
		q1_measure  = (q1, q1_mu)
		
	attach_info = _data_attachment( q1_measure,  xt_measure,  radius_att, attach_mode )
	return [ .01* _Hqp(q, p, radius_def, q_mu = q1_mu) + 1* attach_info[0] , attach_info[1] ]

#================================================================================================

def VisualizationRoutine(Q0, radius_def, q_mu = None) :
	def ShootingVisualization(q,p,grid) :
		return _HamiltonianCarrying(q, p, grid, radius_def, q_mu = q_mu)
	return ShootingVisualization

def perform_matching( Q0, Xt, radius_def = .1, radius_att = .5, 
					  scale_momentum = 1, scale_attach = 1, 
					  attach_mode = "OT",
					  show_trajectories = True,
					  data_type = "curves") :
	"Performs a matching from the source Q0 to the target Xt, returns the optimal momentum P0."
	
	if data_type == "curves" :
		(Xt_x, Xt_mu) = Xt.to_measure()      # Transform the target into a measure once and for all
		transport_action = "image"; Q1_mu = None
		Q0_points   = Q0.points
		connec      = torch.from_numpy(Q0.connectivity).type(dtypeint)
		connec_plot = connec
	elif data_type == "landmarks" :
		(Xt_x, Xt_mu) = Xt
		transport_action = "measure"
		Q1_mu = Variable(torch.from_numpy( Q0[1]   ).type(dtype), requires_grad=False)
		Q0_points   = Q0[0]
		connec      = None
		connec_plot = Q0[1]
		Q0 = Curve(Q0[0], Q0[1]) # for plotting purpose
		Xt = Curve(Xt_x, Xt_mu) # for plotting purpose
		
	# Declaration of variable types -------------------------------------------------------------
	# Cost is a function of 6 parameters :
	# The source 'q',                    the starting momentum 'p',
	# the target points 'xt_x',          the target weights 'xt_mu',
	# the deformation scale 'sigma_def', the attachment scale 'sigma_att'.
	q0    = Variable(torch.from_numpy(    Q0_points ).type(dtype), requires_grad=True)
	p0    = Variable(torch.from_numpy( 0.0*normal(size=Q0_points.shape) ).type(dtype), requires_grad=True )
	Xt_x  = Variable(torch.from_numpy( Xt_x         ).type(dtype), requires_grad=False)
	Xt_mu = Variable(torch.from_numpy( Xt_mu        ).type(dtype), requires_grad=False)
	radius_def = Variable(torch.from_numpy(np.array([radius_def])).type(dtype), requires_grad=False)
	
	
	# Compilation. Depending on settings specified in the ~/.theanorc file or explicitely given
	# at execution time, this will produce CPU or GPU code under the hood.
	def Cost(q,p, xt_x,xt_mu, q1_mu = None) : 
		return _cost( q,p, (xt_x,xt_mu), connec, radius_def, radius_att, attach_mode, q1_mu = q1_mu )
	
	# Display pre-computing ---------------------------------------------------------------------
	g0,cgrid = GridData() ; G0 = Curve(g0, cgrid )
	g0 = Variable( torch.from_numpy( g0 ).type(dtype), requires_grad = False )
	# Given q0, p0 and grid points grid0 , outputs (q1,p1,grid1) after the flow
	# of the geodesic equations from t=0 to t=1 :
	ShootingVisualization = VisualizationRoutine(q0, radius_def, q_mu = Q1_mu) 
	
	# L-BFGS minimization -----------------------------------------------------------------------
	start_time = time.time()
	from scipy.optimize import minimize
	def matching_problem(p0, q1_mu = None) :
		"Energy minimized in the variable 'p0'."
		[c, info] = Cost(q0, p0, Xt_x, Xt_mu, q1_mu = q1_mu)
		elapsed_time = time.time() - start_time
		print("Elapsed time : ", elapsed_time)
		matching_problem.Info = info
		if (matching_problem.it % 10 == 0):# and (c.data.cpu().numpy()[0] < matching_problem.bestc):
			matching_problem.bestc = c.data.cpu().numpy()[0]
			q1,p1,g1 = ShootingVisualization(q0, p0, g0)
			
			q1 = q1.data.cpu().numpy()
			p1 = p1.data.cpu().numpy()
			g1 = g1.data.cpu().numpy()
			
			Q1 = Curve(q1, connec_plot) # N.B.: if landmarks, connec = None
			G1 = Curve(g1, cgrid )
			DisplayShoot( Q0, G0,       p0.data.cpu().numpy(), 
			              Q1, G1, Xt, info.data.cpu().numpy(),
			              matching_problem.it, scale_momentum, scale_attach,
			              attach_mode)
			
			if show_trajectories :
				Qts, Pts, Gts = _HamiltonianTrajectory(q0, p0, g0, radius_def, q_mu = q1_mu)
				
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
	
	optimizer = torch.optim.LBFGS(
					[p0],
					max_iter = 1000, 
					tolerance_change = .000001, 
					history_size = 10)
	#optimizer = torch.optim.Adam(
	#				[p0])
	time1 = time.time()
	def closure():
		optimizer.zero_grad()
		
		if   transport_action == "image" :
			c = matching_problem(p0) # q1_mu will be recomputed at t=1
		elif transport_action == "measure" :
			c = matching_problem(p0, Q1_mu) # q1_mu is fixed, equal to q0_mu
		c.backward()
		return c
	for it in range(100) :
		optimizer.step(closure)
	time2 = time.time()
	return p0, matching_problem.Info

def matching_demo(source_file, target_file, 
                  radius_def, radius_att, 
                  scale_mom = 1, scale_att = 1,
                  attach_mode = "OT") :
	Q0 = Curve.from_file('data/' + source_file) # Load source...
	Xt = Curve.from_file('data/' + target_file) # and target.
	
	# Compute the optimal shooting momentum :
	p0, info = perform_matching( Q0, Xt, 
	                             radius_def, radius_att, 
	                             scale_mom, scale_att,
	                             attach_mode ) 


if __name__ == '__main__' :
	plt.ion()
	plt.show()
	# N.B. : this minimalistic toolbox showcases the Hamiltonian shooting theory...
	#        To get good-looking matching results on a consistent basis, you should
	#        use a data attachment term which takes into account the orientation of curve
	#        elements, such as "currents" and "varifold" kernel-formulas, not implemented here.
	#matching_demo('australopithecus.vtk','sapiens.vtk', (.05,.2), scale_mom = .1,scale_att = .1)
	matching_demo('amoeba_1.png', 'amoeba_2.png', 
	               radius_def = .2, radius_att = 0, 
	               scale_mom  = 4,  scale_att  = 0,
	               attach_mode = "OT")
