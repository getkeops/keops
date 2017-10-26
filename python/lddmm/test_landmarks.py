import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '..')

import numpy as np          # standard array library
from   matching import perform_matching

mode = "2 ellipses"

if mode == "orbit" :
	Q0_x  = np.array([[0,.5], [.5,.3]])
	Q0_mu = [ 1., 1.]
	
	Xt_x  = np.array([[0.1,.5], [-.3,.2]])
	Xt_mu = [ 1., 1.]
elif mode == "switch" :
	Q0_x  = np.array([[0.5,-1.25], [.5,.75]])
	Q0_mu = [ 1., 1.]
	
	Xt_x  = np.array([[0.5,.75], [.5,-1.25]])
	Xt_mu = [ 1., 1.]
elif mode == "4 points" :
	Q0_x  = np.array([[-1.,-1.], [-.45,1.], [.37,1.2], [1.5,-.01]])
	Q0_mu = [ 1., 1., 1., 1.]
	
	Xt_x  = np.array([[-.9,.149], [.23,.23], [.21,-.1], [.958,-.875]])
	Xt_mu = [ 1., 1., 1., 1.]
elif mode == "6 points" :
	Q0_x  = np.array([[-1.,-1.], [-.45,1.], [.37,1.2], [1.5,-.01], [.8,-.9], [-.1,-1.7]])
	Q0_mu = [ 1., 1., 1., 1., 1., 1.]
	
	Xt_x  = np.array([[-.9,.149], [.23,.23], [.21,-.1], [.958,-.875], [-.11,-.854], [-.33,-.143]])
	Xt_mu = [ 1., 1., 1., 1., 1., 1.]
elif mode == "faraday" :
	Q0_x  = np.array([[0,.5], [.5,.3]])
	Xt_x  = np.array([[0.1,.5], [-.3,.2]])
	
	t = np.linspace(0, 2*np.pi, 13)[:-1]
	Q0_x2 = np.vstack( (np.cos(t),        np.sin(t)) ).T
	Xt_x2 = np.vstack( (np.cos(t),np.sin(t)) ).T
	
	Q0_x = np.vstack( (Q0_x, Q0_x2) )
	Xt_x = np.vstack( (Xt_x, Xt_x2) )
	Q0_mu = np.ones((len(Q0_x),))
	Xt_mu = np.ones((len(Xt_x),))
elif mode == "rotation" :
	t = np.linspace(0, 2*np.pi, 1001)[:-1]
	Q0_x = np.vstack( (np.cos(t),        np.sin(t)) ).T
	Xt_x = np.vstack( (np.cos(t+np.pi/2),np.sin(t+np.pi/2)) ).T
	Q0_mu = np.ones((len(Q0_x),))/ len(Q0_x)
	Xt_mu = np.ones((len(Xt_x),))/ len(Xt_x)
elif mode == "translation" :
	t = np.linspace(0, 2*np.pi, 1001)[:-1]
	Q0_x = .5*np.vstack( (np.cos(t)-.6, np.sin(t)) ).T
	Xt_x = .5*np.vstack( (np.cos(t)+.6, np.sin(t)) ).T
	Q0_mu = np.ones((len(Q0_x),)) / len(Q0_x)
	Xt_mu = np.ones((len(Xt_x),)) / len(Xt_x)
elif mode == "ellipses" :
	t = np.linspace(0, 2*np.pi, 501)[:-1]
	Q0_x = .5*np.vstack( (np.cos(t)-.6+ .05*np.cos(2*t), np.sin(t)) ).T
	Xt_x = .5*np.vstack( (np.cos(t)+.6, .5*np.sin(t)) ).T
	Q0_mu = np.ones((len(Q0_x),)) / len(Q0_x)
	Xt_mu = np.ones((len(Xt_x),)) / len(Xt_x)
elif mode == "Cs" :
	t = np.linspace(-np.pi, np.pi, 501)[:-1]
	Q0_x = .5*np.vstack( (np.cos(.2*t), np.sin(.2*t)) ).T
	Xt_x = .5*np.vstack( (np.cos(.8*t), np.sin(.8*t)) ).T
	Q0_mu = np.ones((len(Q0_x),)) / len(Q0_x)
	Xt_mu = np.ones((len(Xt_x),)) / len(Xt_x)
elif mode == "2 ellipses" :
	t = np.linspace(0, 2*np.pi, 501)[:-1]
	Q01_x = .5*np.vstack( (.5*np.cos(t)-.6+ .05*np.cos(2*t),.5*np.sin(t) +  .6) ).T
	Xt1_x = .5*np.vstack( (.5*np.cos(t)+.6,                 .25*np.sin(t) + 1.0) ).T
	
	Q02_x = .5*np.vstack( (.5*np.cos(t)+.2+ .05*np.cos(2*t),.5*np.sin(t) - 1.) ).T
	Xt2_x = .5*np.vstack( (.75*np.cos(t)+.1,                 .25*np.sin(t) - 1.8) ).T
	
	Q0_x = np.vstack( (Q01_x,Q02_x))
	Xt_x = np.vstack( (Xt1_x,Xt2_x))
	
	Q0_mu = np.ones((len(Q0_x),)) / len(Q0_x)
	Xt_mu = np.ones((len(Xt_x),)) / len(Xt_x)

Q0 = [np.array(Q0_x).astype('float32'), np.array(Q0_mu).astype('float32')]
Xt = [np.array(Xt_x).astype('float32'), np.array(Xt_mu).astype('float32')]


if True : # Add some noise to break symmetries?
	from numpy.random import normal
	Q0[0] += normal(loc=0.0, scale=.005, size=Q0[0].shape).astype('float32')
	Xt[0] += normal(loc=0.0, scale=.005, size=Xt[0].shape).astype('float32')
	
	

# Compute the optimal shooting momentum :
p0, info = perform_matching( Q0, Xt, 
							 radius_def      = .15,   radius_att = .2, 
							 scale_momentum  = 50/len(Q0_x),    scale_attach = 1.,
							 attach_mode     = "L2", data_type = "landmarks",
							 show_trajectories = True ) 







