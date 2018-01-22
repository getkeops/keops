
import torch
from torch.autograd import Variable, grad
from .kernel_product import _kernel_product

# Pytorch is a fantastic deep learning library : it transforms symbolic python code
# into highly optimized CPU/GPU binaries, which are called in the background seamlessly.
# It can be thought of as a "heir" to the legacy Theano library (RIP :'-( ).
#
# N.B. : On my Dell laptop, I have a GeForce GTX 960M with 640 Cuda cores and 2Gb of memory.
#
# We now show how to code a whole LDDMM pipeline into one (!!!) page of torch symbolic code.

# Part 1 : cometric on the space of landmarks, kinetic energy on the phase space (Hamiltonian)===

def _K(x, y, p, params, y_mu = None) :
    """
    Operator which links a momentum 'p', supported by a measure 'y',
    to an output velocity field 'v' sampled on a point cloud 'x'.
    Mathematically speaking, this operator is linked to the *cometric*
    of our deformation model.

    The default behavior is to use the standard LDDMM 'kernel' procedure
    which uses the measure 'y' supporting 'p' as a point cloud: the weights
    y_mu are not needed.

    However if normalize=True, we shall use a *normalized kernel*,
    which requires the weights y_mu. This simple switch allows us
    to turn an *extrinsic shape deformation* routine into a *measure
    transportation* program. See our report or e-mail us for complete 
    documentation on this new model (2017-2018, J. Feydy, A. Trouve).

    Args:
        x ( (N,D) torch Variable) : sampling locations (point cloud)
        y ( (M,D) torch Variable) : the point cloud which supports p
        p ( (M,E) torch Variable) : the momentum (vector field)
        params (dict)             : convenient way of storing parameters
        y_mu ( None or (M,) torch Variable) : the weights associated to y
    
    Returns:
        v ( (N,E) torch Variable) : sampled velocity field
    """
    kernel = params ; normalize = params["normalize"]

    if not normalize :
        return _kernel_product(x, y, p, kernel)
    else :
        # We assume that y is the actual measure,
        # and x the point set on which we interpolate
        ly   = Variable( torch.ones((y.size(0),1)).type(p.data.type()) )
        y_mu = y_mu.view(y.size(0),1) # Needed to use pointwise multiplications instead of broadcasting...
        
        for i in range(5) : # Sinkhornization loop. We could use a break statement here...
            ly = torch.sqrt( ly / _kernel_product(y, y, ly * y_mu, kernel) )
        return    _kernel_product(x, y, ly * p,    kernel) \
                / _kernel_product(x, y, ly * y_mu, kernel) 

def _Hqp(q, p, params, q_mu = None) :
    """
    The Hamiltonian, or kinetic energy of the shape q with momentum p.

    Args:
        q ( (N,D) torch Variable) : the deformed shape (point cloud)
        p ( (N,D) torch Variable) : the momentum (vector field supported by q)
        q_mu (None or (N,) torch Variable) : weights associated to the point cloud q

    """
    pKqp =  torch.dot(p.view(-1), _K(q, q, p, params, y_mu = q_mu).view(-1))
    return .5 * pKqp   # typically, $H(q,p) = \frac{1}{2} * sum_{i,j} K_ij p_i.p_j$


# Part 2 : Geodesic shooting ====================================================================
# The partial derivatives of the Hamiltonian are automatically computed !
def _dq_Hqp(q,p, params, q_mu = None) : 
    return grad(_Hqp(q,p, params, q_mu), q, create_graph=True)[0]

def _dp_Hqp(q,p, params, q_mu = None) :
    return grad(_Hqp(q,p, params, q_mu), p, create_graph=True)[0]

def _hamiltonian_step(q,p, params, q_mu = None) :
    "Simplistic euler scheme step with dt = .1."
    return (q + .1 * _dp_Hqp(q,p, params,q_mu) ,
            p - .1 * _dq_Hqp(q,p, params,q_mu) )

def _HamiltonianShooting(q,p, params, q_mu = None) :
    """
    Shoots to time 1 a k-geodesic starting (at time 0) from q with momentum p.
    This routine allows us to encode a *Riemannian prior* in our shape deformation models,
    since Hamilton's equation embodies the notion of geodesic.

    Args:
        q ( (N,D) torch Variable) : the initial point cloud "q_0"
        p ( (N,D) torch Variable) : the initial momentum "p_0" (vector field supported by q) 
        params (dict)             : convenient way of storing parameters
        q_mu ( None or (M,) torch Variable) : the weights associated to q,
                                              needed if params["normalize"]==True
    Returns:
        a pair (q_1, p_1), which has the same shape as (q,p) and corresponds
        to the flow at time 1 of Hamilton's dynamical system, starting
        at time 0 from the point [q,p] in phase space.
    """
    for t in range(10) :
        q,p = _hamiltonian_step(q,p, params,q_mu) # Let's hardcode the "dt = .1"
    return (q,p)                             # and only return the final state + momentum



# Part 2bis : Geodesic shooting + deformation of the ambient space, for visualization ===========
def _HamiltonianCarrying(q, p, g, params, q_mu = None, trajectory = False, endtime = 1.) :
    """
    Similar to _HamiltonianShooting, but also conveys information about the deformation of
    an arbitrary point cloud 'g' (like 'grid') in the ambient space.
    This routine can be used to visualize the underlying diffeomorphism associated
    to a finite-dimensional geodesic [q_t,p_t] (if 'g' has been generated
    using a meshgrid-like routine), or to use a 'control points'-like model.

    Args:
        q ( (N,D) torch Variable) : the initial point cloud "q_0"
        p ( (N,D) torch Variable) : the initial momentum "p_0" (vector field supported by q) 
        g ( (M,D) torch Variable) : the initial *passive* point cloud "g_0"
        params (dict)             : convenient way of storing parameters
        q_mu ( None or (M,) torch Variable) : the weights associated to q,
                                              needed if params["normalize"]==True
        trajectory (bool)         : should we return the eventual state (q_1,p_1,g_1),
                                    or full, sampled trajectories?
        endtime (float)           : end-time (default=1); can be used to visualize "shape extrapolation".

    Returns:
        if traj==False: a triplet of torch Variables (q_1,p_1,g_1)
        if traj==True:  a triplet of lists of torch Variables ([q_0,...,q_1], [p_0,...], [g_0,...])

    """ 
    if trajectory :
        qs = [q] ; ps = [p]; gs = [g]
    for t in range(int(endtime * 10)) : # Let's hardcode the "dt = .1"
        q,p,g = [q + .1 * _dp_Hqp(   q, p, params, q_mu), # = q + .1*_K(q,q,p,...)
                 p - .1 * _dq_Hqp(   q, p, params, q_mu), 
                 g + .1 *      _K(g, q, p, params, q_mu)]
        if trajectory :
            qs.append(q) ; ps.append(p) ; gs.append(g)
    
    if trajectory :
        return qs,ps,gs         # return the states + momenta + grids
    else :
        return q, p, g          # return the final state + momentum + grid
