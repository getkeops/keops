
import numpy as np
import torch
from   torch.autograd import Variable
from   .kernel_product import _kernel_product, Kernel


# L2 DISTANCE (for testing purposes) ==========================================================

def _L2_distance(Mu, Nu, params, info = False) :
	cost = torch.sum( ( (Mu[1] - Nu[1])**2 ) * Mu[0].view(-1,1) )
	return cost, None



# KERNEL DISTANCES ============================================================================


def _kernel_scalar_product(Mu, Nu, params) :
    """
    Computes the kernel scalar product
    <Mu,Nu>_k = < Mu, k \star Nu >                    (convolution product)
              = \sum_{i,j} k(x_i-y_j) * mu_i * nu_j
    The kernel function is specified by "params"

    Args:
        Mu (pair of torch Variables) : = (mu,x) where 'mu' is an (N,)  torch Variable
                                                  and 'x'  is an (N,D) torch Variable
        Nu (pair of torch Variables) : = (nu,y) where 'nu' is an (M,)  torch Variable
                                                  and 'y'  is an (M,D) torch Variable
        params                       : a convenient way of specifying a kernel function

    N.B.: if params specifies a "points+orientations" kernel 
         (say, params["name"]=="gaussian_current"), x and y may instead
         be *pairs* (or even n-uples) of torch Variables.
    """
    (mu, x) = Mu ; (nu, y) = Nu
    k_nu = _kernel_product(x,y,nu.view(-1,1),params)
    return torch.dot( mu.view(-1), k_nu.view(-1) ) # PyTorch syntax for the L2 scalar product...

def _kernel_distance(Mu, Nu, params, info = False) :
    """
    Hilbertian kernel (squared) distance between measures Mu and Nu,
    computed using the fact that
    
    |Mu-Nu|^2_k  =  <Mu,Mu>_k - 2 <Mu,Nu>_k + <Nu,Nu>_k
    
    If "info" is required, we output the values of
         k \star (Mu-Nu)  sampled on a uniform grid,
    to be plotted later.
    
    Strictly speaking, it would make more sense to display
         g \star (Mu-Nu)     where     g \star g = k
    as we would then have
          |Mu-Nu|^2_k  =  |g \star (Mu-Nu)|^2_{L^2}.
        
    But this is easy only for Gaussians...
    """
    D2 =   (   _kernel_scalar_product(Mu,Mu,params) \
           +   _kernel_scalar_product(Nu,Nu,params) \
           - 2*_kernel_scalar_product(Mu,Nu,params) )
    
    kernel_heatmap = None
    if False :#info :
        # Create a uniform grid on the [-2,+2]x[-2,+2] square:
        xmin,xmax,res  = params.get("kernel_heatmap_range", (-2,2,100))
        ticks  = np.linspace( xmin, xmax, res + 1)[:-1] + 1/(2*res) 
        X,Y    = np.meshgrid( ticks, ticks )

        dtype = Mu[0].data.type()
        points = Variable(torch.from_numpy(np.vstack( (X.ravel(), Y.ravel()) ).T).contiguous().type(dtype), requires_grad=False)
        
        # Sample "k \star (Mu-Nu)" on this grid:
        kernel_heatmap   = _kernel_product(points, Mu[1], Mu[0].view(-1,1), params) \
                         - _kernel_product(points, Nu[1], Nu[0].view(-1,1), params)
        kernel_heatmap   = kernel_heatmap.view(res,res) # reshape as a "background" image

    return D2, kernel_heatmap



# OPTIMAL TRANSPORT DISTANCES ========================================================================


def _sinkhorn_loop(Mu, Nu, params) :

    # Extract the parameters from params:
    # N.B.: The user should be aware that this routine solves a *regularized*
    #       OT problem, so we do not provide default value for epsilon.
    eps    = params["epsilon"]  
    kernel = params.get("kernel", {"id"     : Kernel("gaussian(x,y)") ,
                                   "gamma"  :  1 / eps,
                                   "backend": "auto"   } )

    rho    = params.get("rho",  -1)    # Use unbalanced transport?
    tau    = params.get("tau",  0.)    # Use inter/extra-polation?
    nits   = params.get("nits", 1000)  # When shall we stop?
    tol    = params.get("tol",  1e-5)  # When shall we stop?

    # Compute the exponent for the unbalanced Optimal Transport update
    lam = 1. if rho < 0 else rho / (rho + eps)

    # precompute the log-weights, as column vectors
    log_mu  = Mu[0].log().view(-1,1)
    log_nu  = Nu[0].log().view(-1,1)

    # Initialize the log variables
    U   = Variable(torch.zeros_like(log_mu.data))
    V   = Variable(torch.zeros_like(log_nu.data))

    for it in range(nits) : # The Sinkhorn loop, implemented in the log-domain
        U_prev = U          # Store the previous result for the break statement

        # Kernel products + pointwise divisions, combined with an extrapolating scheme if tau<0
        # Mathematically speaking, we're alternating Kullback-Leibler projections.
        V = tau*V + (1-tau)*lam*( log_nu - _kernel_product(Nu[1], Mu[1], U, kernel, mode="log") )
        U = tau*U + (1-tau)*lam*( log_mu - _kernel_product(Mu[1], Nu[1], V, kernel, mode="log") ) 

        # Compute the L1 norm of the update wrt. U. If it's small enough... break the loop!
        err = (eps * (U-U_prev).abs().mean()).data.cpu().numpy()
        if err < tol : break
    
    # Straightforward expression of the dual cost:
    D2 = eps * ( torch.dot(Mu[0].view(-1), U.view(-1)) \
               + torch.dot(Nu[0].view(-1), V.view(-1)) )
    
    # Return the cost alongside the dual variables, as they may come handy
    return D2, U, V

def _wasserstein_distance(Mu, Nu, params, info = False) :
    """
    Log-domain implementation of the Sinkhorn algorithm,
    provided for numerical stability.
    The "multiplicative" standard implementation is replaced
    by an "additive" logarithmic one, as:
    - A is replaced by U_i = log(A_i) = (u_i/eps - .5)
    - B is replaced by V_j = log(B_j) = (v_j/eps - .5)
    - K_ij is replaced by C_ij = - eps * log(K_ij)
                               = |X_i-Y_j|^2
    (remember that epsilon = eps = s^2)
    
    The update step:
    
    " a_i = mu_i / \sum_j k(x_i,y_j) b_j "
    
    is thus replaced, applying log(...) on both sides, by
    
    " u_i = log(mu_i) - log(sum( exp(-C_ij/eps) * exp(V_j) )) ] "
    
    N.B.: By default, we use a slight extrapolation to let the algorithm converge faster.
          As this may let our algorithm diverge... Please set tau=0 to fall back on the
          standard Sinkhorn algorithm.
    """

    D2, U, V = _sinkhorn_loop(Mu, Nu, params)
    
    transport_plan = None
    if info :
        mode = params.get("transport_plan", "none")
        if mode   == "none" :
            None
        elif mode == "minimal" :
            None
        elif mode == "full" :
            eps   = params["epsilon"]
            C = ((Mu[1].unsqueeze(1) - Nu[1].unsqueeze(0) )**2).sum(2)
            transport_plan = ( U.view(-1,1)+V.view(1,-1) - C/eps ).exp()
        else :
            raise ValueError('params["transport_plan"] has incorrect value : ' \
                             + str(params["transport_plan"]) + ".\nCorrect values are " \
                             + '"none", "minimal" and "full".'                               )
    return D2, transport_plan

def _sinkhorn_distance(Mu, Nu, params, info = False) :
    """
    Inspired by "Learning Generative Models with Sinkhorn Divergences"
    by Genevay, Peyré and Cuturi (2017, arxiv.org/abs/1706.00292), we carelessly compute
    a loss function using only a handful of sinkhorn iterations. 
    The expression below is designed to give "relevant" results even
    when the regularization parameter 'params["epsilon"]' is strong 
    or when the Sinkhorn loop has not fully converged.
    
    This formula uses the "dual" Sinkhorn cost and has not been documented anywhere: 
    it is a mere experiment, a compromise between the mathematical theory of OT
    and algorithmic efficiency. As such, it lacks a proper interpretation
    and we prefer not to output any misleading 'informative' transport plan.
    """
    Loss = 2*_sinkhorn_loop(Mu,Nu, params)[0] \
           - _sinkhorn_loop(Mu,Mu, params)[0] - _sinkhorn_loop(Nu,Nu, params)[0]
    transport_plan = None
    return Loss, transport_plan



# CONVENIENCE WRAPPER ====================================================================

def _data_attachment(source, target, params, info=False) :
    """Given two shapes and a dict of parameters, returns a cost."""

    embedding = params.get("features", "locations")
    if   embedding == "locations" :                    # one dirac = one vector x_i or y_j
        Mu = source.to_measure()
        Nu = target.to_measure()
    elif embedding == "locations+directions" :         # one dirac = (x_i,u_i)     or (y_j,v_j)
        Mu = source.to_varifold()                      # N.B.: u_i and v_j 's norms are equal to 1 !
        Nu = target.to_varifold()
    elif embedding == "locations+directions+values" :  # one dirac = (x_i,u_i,s_i) or (y_j,v_j,t_j)
        Mu = source.to_fvarifold() # "functional varifolds": (terminology 
        Nu = target.to_fvarifold() # introduced in the PhD thesis of Nicolas Charon)
    else :
        raise NotImplementedError('Unknown features type : "'+embedding+'". ' \
                                  'Available values are "locations" and "locations+directions".')

    attachment_type = params["formula"]
    routines = { "L2"          : _L2_distance,
                 "kernel"      : _kernel_distance,
                 "wasserstein" : _wasserstein_distance,
                 "sinkhorn"    : _sinkhorn_distance    }

    if   attachment_type in routines :
        return routines[attachment_type](Mu, Nu, params, info)
    else :
        raise NotImplementedError('Data attachment formula not supported: "'+attachment_type+'". ' \
                                 +'Correct values : "' + '", "'.join(routines.keys()) + '".')
