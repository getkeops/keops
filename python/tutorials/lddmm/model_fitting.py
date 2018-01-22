
import os
import torch
import matplotlib.pyplot as plt
from math import isnan

def FitModel(params, Model, target) :
    """
    """
    par_optim =    params.get("optimization", {}   )
    nits      = par_optim.get("nits",         100  )
    nlogs     = par_optim.get("nlogs",        10   )
    tol       = par_optim.get("tol",          1e-7 )
    method    = par_optim.get("method",       "Adam" )

    # We'll minimize the model's cost
    # with respect to the model's parameters using a standard gradient-like
    # descent scheme. As we do not perform any kind of line search, 
    # this algorithm may diverge if the learning rate is too large !
    # For robust optimization routines, you may consider using
    # the scipy.optimize API with a "parameters <-> float64 vector" wrapper.
    if method == "Adam" :
        lr  = par_optim.get("lr",  .1 )
        eps = par_optim.get("eps", .01)
        optimizer = torch.optim.Adam(Model.parameters(), lr=lr, eps=eps)
    elif method == "L-BFGS" :
        None
    else :
        raise NotImplementedError('Optimization method not supported : "'+method+'". '\
                                  'Available values are "Adam" and "L-BFGS".')

    # We'll plot results on-the-fly, and save the list of costs across iterations
    if "display" in params :
        fig_model = plt.figure(figsize=(10,10), dpi=100) ; ax_model  = plt.subplot(1,1,1) ; ax_model.autoscale(tight=True)
        fig_costs = plt.figure(figsize=(10,10), dpi=100) ; ax_costs  = plt.subplot(1,1,1) ; ax_costs.autoscale(tight=True)
    costs = []
    
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
        
        # Break the loop if cost is NaN, or if the cost's variation is below the tolerance param:
        if  isnan(costs[-1]) or ( len(costs)>1 and abs(costs[-1]-costs[-2]) < tol ) : 
            FitModel.breakloop = True

        if it % nlogs == 0: # Display the current model ----------------------------------------
            print("Iteration ",i,", Cost = ", costs[-1])

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
    
    for i in range(nits+1) :        # Fixed number of iterations
        optimizer.step(closure)     # "Gradient descent" step.
        if FitModel.breakloop : break
            