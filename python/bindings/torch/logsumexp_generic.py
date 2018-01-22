import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '..')

from cudaconv import cuda_conv_generic
import torch
import numpy
from torch.autograd import Variable

from .kernel_product_generic import GenericKernelProduct

# See github.com/pytorch/pytorch/pull/1016 , pytorch.org/docs/0.2.0/notes/extending.html
# for reference on the forward-backward syntax
class GenericLogSumExp(torch.autograd.Function):
	""" 
	Computes a Generic LogSumExp specified by a formula (string) such as the Gaussian kernel:
	formula = " ScalProduct< Scal<Constant<C>, Minus<SqNorm2<Subtract<X,Y>>> > ,  B> "
	which will be turned into
	formula = "LogSumExp<" + formula + ">"
	"""
	
	@staticmethod
	def forward(ctx, backend, aliases, formula, signature, sum_index, *args):
		""" 
		"""
		# Save everything to compute the gradient -----------------------------------------------
		# N.B.: relying on the "ctx.saved_variables" attribute is necessary
		#       if you want to be able to differentiate the output of the backward
		#       once again. It helps pytorch to keep track of "who is who".
		ctx.aliases   = aliases
		ctx.formula   = formula
		ctx.signature = signature
		ctx.sum_index = sum_index
		ctx.backend   = backend
		
		# Get the size nx by looping on the signature until we've found an "x_i" ----------------
		n = -1
		for ( index, sig) in enumerate(signature[1:]) : # Omit the output
			if sig[1] == sum_index :
				n = len( args[index] ) # Lengths compatibility is done by cuda_conv_generic
				break
		if n == -1 and sum_index == 0: raise ValueError("The signature should contain at least one indexing argument x_i.")
		if n == -1 and sum_index == 1: raise ValueError("The signature should contain at least one indexing argument y_j.")
		
		# Conversions between the "(m,s)" (2 scalars) output format of the logsumexp cuda routines
		# and the "m + log(s)" (1 scalar) format of the pytorch wrapper
		if not signature[0][0] == 1 : raise ValueError("LogSumExp has only been implemented for scalar-valued formulas.")
		
		# in the backend, logsumexp results are encoded as pairs of real numbers:
		signature = signature.copy()
		signature[0] = (2, signature[0][1])

		formula = "LogSumExp<"+formula+">"

		# Actual computation --------------------------------------------------------------------
		result  = torch.zeros( n,  signature[0][0] ).type(args[0].type())  # Init the output of the convolution
		cuda_conv_generic(formula, signature, result, *args,               # Inplace CUDA routine
		                  backend   = backend,
		                  aliases   = aliases, sum_index   = sum_index,
		                  cuda_type = "float", grid_scheme = "2D") 
		result  = result.view( n, signature[0][0] )

		# (m,s) represents exp(m)*s, so that "log((m,s)) = log(exp(m)*s) = m + log(s)"
		result  = result[:,0] + result[:,1].log()
		result  = result.view(-1,1)

		ctx.save_for_backward( *(args+(result,)) ) # Call at most once in the "forward"; we'll need the result!
		return result
	
	@staticmethod
	def backward(ctx, G):
		"""
		"""
		aliases   = ctx.aliases
		formula   = ctx.formula
		signature = ctx.signature
		sum_index = ctx.sum_index
		backend   = ctx.backend
		args      = ctx.saved_variables # Unwrap the saved variables
		result    = args[ -1]
		args      = args[:-1]

		#print(args)
		#print(result)
		#print(result.grad_fn)

		# Compute the number of arguments which are not parameters
		nvars = 0 ; 
		for sig in signature[1:] : 
			if sig[1] != 2 : nvars += 1
		
		# If formula takes 5 variables (numbered from 0 to 4), then:
		# - the previous output should be given as a 6-th variable (numbered 5), 
		# - the gradient wrt. the output, G, should be given as a 7-th variable (numbered 6),
		# both with the same dim-cat as the formula's output. 
		res     = "Var<"+str(nvars)  +","+str(signature[0][0])+","+str(signature[0][1])+">"
		eta     = "Var<"+str(nvars+1)+","+str(signature[0][0])+","+str(signature[0][1])+">"
		grads   = []                # list of gradients wrt. args;
		arg_ind = 3; var_ind = -1;  # current arg index (3 since aliases, ... are in front of the tensors); current Variable index;
		
		for sig in signature[1:] : # Run through the actual parameters, given in *args in the forward.
			arg_ind += 1
			if sig[1] == 2 : # we're referring to a parameter
				grads.append(None) # Not implemented yet
			else :
				var_ind += 1 # increment the Variable count
				if not ctx.needs_input_grad[arg_ind] :  # If the current gradient is to be discarded immediatly...
					grads.append(None)                  # Don't waste time computing it.
				else :                                  # Otherwise, the current gradient is really needed by the user:
					# adding new aliases is waaaaay too dangerous if we want to compute
					# second derivatives, etc. So we make explicit references to Var<ind,dim,cat> instead.
					var         = "Var<"+str(var_ind)+","+str(sig[0])+","+str(sig[1])+">" # V
					formula_g   = "Grad< "+ formula +","+ var +","+ eta +">"              # Grad<F,V,G>
					formula_g   = "Scal<Exp<Subtract<" + formula + "," + res + ">>," + formula_g + ">"
					signature_g = [ sig ] + signature[1:] + signature[:1] + signature[:1]
					# sumindex is "the index that stays in the end", not "the one in the sum" 
					# (It's ambiguous, I know... But it's the convention chosen by Joan, which makes
					#  sense if we were to expand our model to 3D tensors or whatever.)
					sumindex_g  = sig[1]     # The sum will be "eventually indexed just like V".
					args_g      = args + (result,G) # Don't forget the value & gradient to backprop !
					
					# N.B.: if I understand PyTorch's doc, we should redefine this function every time we use it?
					genconv = GenericKernelProduct().apply  
					grads.append( genconv( backend, aliases, formula_g, signature_g, sumindex_g, *args_g )  )
		
		# Grads wrt.  backend, aliases, formula, signature, sum_index, *args
		return      (   None,   None,    None,      None,      None, *grads )


if __name__ == "__main__":
	
	# Computation are made in float32
	dtype = torch.FloatTensor 
	
	backend = "libkp" # Other value : 'pytorch'
	
	if   backend == "libkp" :
		def kernel_product_log(s, x, y, v, kernel) :
			genconv  = GenericLogSumExp().apply
			dimpoint = x.size(1) ; dimout = b.size(1)
			
			if not dimout == 1 : 
				raise ValueError("As of today, LogSumExp has only been implemented for scalar-valued functions.")
			if True :
				aliases  = ["DIMPOINT = "+str(dimpoint), "DIMOUT = "+str(dimout),
						    "C = Param<0>"          ,   # 1st parameter
						    "X = Var<0,DIMPOINT,0>" ,   # 1st variable, dim DIM,    indexed by i
						    "Y = Var<1,DIMPOINT,1>" ,   # 2nd variable, dim DIM,    indexed by j
						    "V = Var<2,DIMOUT  ,1>" ]   # 3rd variable, dim DIMOUT, indexed by j
						   
				# stands for:     R_i   ,   C  ,      X_i    ,      Y_j    ,     V_j    .
				signature = [ (dimout,0), (1,2), (dimpoint,0), (dimpoint,1), (dimout,1) ]
				
				#   R   =                     C *   -          |         X-Y|^2   +  V
				formula = "Add< Scal<Constant<C>, Minus<SqNorm2<Subtract<X,Y>>> > ,  V>"
			
			else :
				aliases = []
				C = "Param<0>"
				X = "Var<0,"+str(dimpoint)+",0>"
				Y = "Var<1,"+str(dimpoint)+",1>"
				V = "Var<2,"+str(dimout  )+",1>"
				formula = "Add< Scal<Constant<"+C+">, Minus<SqNorm2<Subtract<"+X+","+Y+">>> >,  "+V+">"
			
			sum_index = 0 # the output vector is indexed by "i" (CAT=0)
			return genconv( aliases, formula, signature, sum_index, 1/(s**2), x, y, v )
		
	elif backend == "pytorch" :
		def kernel_product_log(s, x, y, v, kernel) :
			x_col = x.unsqueeze(1) # (N,D) -> (N,1,D)
			y_lin = y.unsqueeze(0) # (N,D) -> (1,N,D)
			sq    = torch.sum( (x_col - y_lin)**2 , 2 )
			K     =  -sq / (s**2) + v.view(1,-1)
			return K.exp().sum(1).log().view(-1,1)
			
			
			
	#--------------------------------------------------#
	# Init variables to get a minimal working example:
	#--------------------------------------------------#
	
	N = 10000 ; M = 10000 ; D = 3 ; E = 1
	
	e = .6 * torch.linspace(  0, 5,N*D).type(dtype).view(N,D)
	e = torch.autograd.Variable(e, requires_grad = True)
	
	a = .6 * torch.linspace(  0, 5,N*E).type(dtype).view(N,E)
	a = torch.autograd.Variable(a, requires_grad = True)
	
	x = .6 * torch.linspace(  0, 5,N*D).type(dtype).view(N,D)
	x = torch.autograd.Variable(x, requires_grad = True)
	
	y = .2 * torch.linspace(  0, 5,M*D).type(dtype).view(M,D)
	y = torch.autograd.Variable(y, requires_grad = True)
	
	b = .6 * torch.linspace(-.2,.2,M*E).type(dtype).view(M,E)
	b = torch.autograd.Variable(b, requires_grad = True)
	
	s = torch.Tensor([2.5]).type(dtype)
	s = torch.autograd.Variable(s, requires_grad = False)
	
	#--------------------------------------------------#
	# check the class KernelProduct
	#--------------------------------------------------#
	Kyy_b = kernel_product_log(s,y,y,b, "gaussian")
	print("kernel product (log) ...: \n", Kyy_b[:10,:].data.cpu().numpy())

	if True :
		
		H = (Kyy_b**2).sum()
		
		print('----------------------------------------------------')
		print("H :") ; print(H)
	
	if True :
		grad_y  = torch.autograd.grad(H,y,create_graph = True)[0]
		print('grad_y...   :\n', grad_y[:10,:].data.cpu().numpy())
		grad_b  = torch.autograd.grad(H,b,create_graph = True)[0]
		print('grad_b...   :\n', grad_b[:10,:].data.cpu().numpy())

	if True :
		dummy = torch.linspace(0,1,grad_y.numel()).type(dtype).view(grad_y.size())
		grad_yy = torch.autograd.grad(grad_y,y, dummy, create_graph = True)[0]
		print('grad_yy...  :\n', grad_yy[:10,:].data.cpu().numpy())


	if False :
		def to_check( X, Y, B ):
			return kernel_product_log(s, X, Y, B, "gaussian")

		gc = torch.autograd.gradcheck(to_check, inputs=(x, y, b) , eps=1e-3, atol=1e-3, rtol=1e-3 )
		print('Gradcheck for Hamiltonian: ',gc)
		print('\n')
