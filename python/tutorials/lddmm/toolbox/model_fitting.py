
import os
import torch
import matplotlib.pyplot as plt
from math import isnan
import numpy as np
from scipy.optimize import minimize

import warnings
warnings.filterwarnings("ignore",".*GUI is implemented.*") # annoying warning with pyplot and pause...

def model_to_numpy(model, grad=False) :
    """
    The fortran routines used by scipy.optimize expect float64 vectors
    instead of the gpu-friendly float32 matrices: we need conversion routines.
    """
    if not all( param.is_contiguous() for param in model.parameters() ) :
        raise ValueError("Scipy optimization routines are only compatible with parameters given as *contiguous* tensors.")

    if grad :
        tensors = [param.grad.data.view(-1).cpu().numpy() for param in model.parameters()]
    else :
        tensors = [param.data.view(-1).cpu().numpy()      for param in model.parameters()]
    return np.ascontiguousarray( np.hstack(tensors) , dtype='float64' )

def numpy_to_model(model, vec) :
    i = 0
    for param in model.parameters() :
        offset = param.numel()
        param.data = torch.from_numpy(vec[i:i+offset]).view(param.data.size()).type(param.data.type())
        i += offset

    if i != len(vec) :
        raise ValueError("The total number of variables in model is not the same as in 'vec'.")

def FitModel(params, Model, target) :
    """
    """

    # Load parameters =====================================================================================================
    par_optim =    params.get("optimization", {}   )
    nits      = par_optim.get("nits",         100  )
    nlogs     = par_optim.get("nlogs",        10   )
    tol       = par_optim.get("tol",          1e-7 )
    method    = par_optim.get("method",       "L-BFGS" )

    # We'll minimize the model's cost
    # with respect to the model's parameters using a standard gradient-like
    # descent scheme. As we do not perform any kind of line search, 
    # this algorithm may diverge if the learning rate is too large !
    # For robust optimization routines, you may consider using
    # the scipy.optimize API with a "parameters <-> float64 vector" wrapper.
    use_scipy = False
    if method == "Adam" :
        lr  = par_optim.get("lr",  .1 )
        eps = par_optim.get("eps", .01)
        optimizer = torch.optim.Adam(Model.parameters(), lr=lr, eps=eps)
    elif method == "L-BFGS" :
        optimizer = torch.optim.SGD(Model.parameters(), lr=1.) # We'll just use its "zero_grad" method...

        lr        = par_optim.get("lr",     .1 )
        maxcor    = par_optim.get("maxcor", 10 )
        use_scipy = True
        method    = 'L-BFGS-B'
        options   = dict( maxiter = nits,
                          ftol    = tol,          # Don't bother fitting the shapes to float precision
                          maxcor  = maxcor        # Number of previous gradients used to approximate the Hessian
                    )
    else :
        raise NotImplementedError('Optimization method not supported : "'+method+'". '\
                                  'Available values are "Adam" and "L-BFGS".')

    # We'll plot results on-the-fly, and save the list of costs across iterations =========================================
    if "display" in params :
        fig_model = plt.figure(figsize=(10,10), dpi=100) ; ax_model  = plt.subplot(1,1,1) ; ax_model.autoscale(tight=True)
        fig_costs = plt.figure(figsize=(10,10), dpi=100) ; ax_costs  = plt.subplot(1,1,1) ; ax_costs.autoscale(tight=True)
    costs = []
    

    # Define the "closures" associated to our model =======================================================================

    FitModel.nit = -1 ; FitModel.breakloop = False
    def closure():
        """
        Encapsulates a problem + display iteration into a single callable statement.
        This wrapper is needed if you choose to use LBFGS-like algorithms, which
        (should) implement a careful line search along the gradient's direction.
        """
        FitModel.nit += 1 ; it = FitModel.nit
        # Minimization loop --------------------------------------------------------------------
        optimizer.zero_grad()                      # Reset the gradients (PyTorch syntax...).
        cost = Model.cost(params, target)[0]
        costs.append(cost.data.cpu().numpy()[0])   # Store the "cost" for plotting.
        cost.backward(retain_graph=True)           # Backpropagate to compute the gradient.
        
        # Break the loop if the cost's variation is below the tolerance param:
        if ( len(costs)>1 and abs(costs[-1]-costs[-2]) < tol ) :
            FitModel.breakloop = True

        if it % nlogs == 0: # Display the current model ----------------------------------------
            print("Iteration ",it,", Cost = ", costs[-1])

            if "display" in params : # Real-time display:
                ax_model.clear()
                Model.plot(ax_model, params, target)
                ax_model.axis(params["display"]["limits"]) ; ax_model.set_aspect('equal') ; plt.draw() ; plt.pause(0.01)

                if "save" in params :
                    screenshot_filename = params["save"]["output_directory"]+"descent/plot_"+str(it)+'.png'
                    os.makedirs(os.path.dirname(screenshot_filename), exist_ok=True)
                    fig_model.savefig( screenshot_filename )
                    
            if "save" in params : # Save for later use:
                Model.save(params, target, it=it)
        return cost
    
    # Scipy-friendly wrapper ------------------------------------------------------------------------------------------------
    def numpy_closure(vec) :
        """
        Wraps the PyTorch closure into a 'float64'-vector routine,
        as expected by scipy.optimize.
        """
        vec    = lr * vec.astype('float64')            # scale the vector, and make sure it's given as float64
        numpy_to_model(Model, vec)                     # load this info into Model's parameters
        c      = closure().data.cpu().numpy()[0]       # compute the cost and accumulate the gradients wrt. the parameters
        dvec_c = lr * model_to_numpy(Model, grad=True) # -> return this gradient, as a properly rescaled numpy vector
        return (c, dvec_c)

    # Actual minimization loop ===============================================================================================
    if use_scipy :
        res = minimize( numpy_closure,      # function to minimize
                model_to_numpy(Model), # starting estimate
                method  = method,
                jac     = True,             # matching_problems also returns the gradient
                options = options    )
    else :
        for i in range(nits+1) :            # Fixed number of iterations
            optimizer.step(closure)         # "Gradient descent" step.
            if FitModel.breakloop : break
            