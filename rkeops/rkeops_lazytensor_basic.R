library(rkeops)

set_rkeops_option("tagCpuGpu", 0)
set_rkeops_option("precision", "double")


# Minimal LazyTensor implementation. LazyTensors objects are wrappers around
# R matrices or vectors that are used to create symbolic formulas for the KeOps
# reduction operations. A typical use case is the following
#    x = runif(100,3)          # arbitrary R matrix representing 100 data points in R^3
#    y = runif(150,3)          # arbitrary R matrix representing 150 data points in R^3
#    s = 0.1                   # scale parameter
#    x_i = LazyTensor(x,"i")   # symbolic object representing an arbitrary row of x, indexed by the letter "i"
#    y_j = LazyTensor(y,"j")   # symbolic object representing an arbitrary row of y, indexed by the letter "j"
#    D_ij = sum((x_i-y_j)^2)   # symbolic matrix of pairwise squared distances, with 100 rows and 150 columns
#    K_ij = exp(-D_ij/s^2)     # symbolic matrix, 100 rows and 150 columns
#    res = sum(K_ij,index="i") # actual R matrix (in fact a row vector of length 150 here) containing the column sums of K_ij (i.e. the sums over the "i" index, for each "j" index)
#
# How does LazyTensor class work internally ?
# LazyTensor objects mainly contain two members:
#  - formula : a string defining the mathematical operation to be computed by the KeOps routine. 
#  - vars : a list of R matrices which will be the inputs of the KeOps routine
# Let us detail these two members in the previous example :
#  x_i : formula="Var(0,3,0)" and vars = { x }
#  y_j : formula="Var(0,3,1)" and vars = { y }
#  D_ij : formula="Sum(Square(Var(0,3,0)-Var(1,3,1)))" and vars = {x,y}
#  K_ij : formula="Exp(-Sum(Square(Var(0,3,0)-Var(1,3,1)))/Var(2,1,1))" and vars = {x,y,s^2}
#  Then the last command sum(K_ij,index="i") 
#  builds up the final formula string "Sum_Reduction(Exp(-Sum(Square(Var(0,3,0)-Var(1,3,1)))/Var(2,1,1)),1)"
#  and passes it to the keops_kernel function. The resulting R routine is evaluated with inputs x, y, and s^2.


# Here we define only a small set of operations, needed to run the small
# example below. This is done using S3 classes, but maybe it would be better and
# cleaner to use Reference classes...

# Function LazyTensor : The entry point for LazyTensor : LazyTensor(x,index) will turn a R matrix or R vector
# x into a LazyTensor object with some attached index="i" or index="j". More precisely,
# the three cases of use are :
#  -if x is a R matrix of size N*D:
#     LazyTensor(x,"i")
#     LazyTensor(x,"j")
#  In these cases the output object should be understood as an arbitrary row of x, indexed by "i" or "j"
#  -if x is a R vector of size D:
#     LazyTensor(x)
#  In this case the output object corresponds to a parameter vector, without any attached index
LazyTensor = function(x,index)
{
    ni = 0   # will correpond to the number of rows of the input if it is an "i" indexed variable
    nj = 0   # will correpond to the number of rows of the input if it is a "j" indexed variable

    # 1) input is a matrix, treated as indexed variable, so index must be "i" or "j"
    if(is.matrix(x))
    {
	x = t(x)          # we transpose the input matrix x because KeOps needs C-contiguous arrays
        d = nrow(x)       # d is the dimension, now the number of rows of x
        if(index=='i')
        {
            cat = 0       # cat is the KeOps "category", 0, 1, or 2, corresponding to the 3 different cases of use (see above)
            ni = ncol(x)
        }
        else
        {
            cat = 1
            nj = ncol(x)
        }
    }
    # 2) else we assume x is a numeric vector, treated as parameter, then converted to matrix
    else
    {
        d = length(x)
        cat = 2
        x = t(matrix(x))   # now x is a row matrix, which is the correct shape for KeOps routines
    }

    # Now we define "formula", a string specifying the variable for KeOps C++ codes.
    formula = paste('Var(0,',d,',',cat,')', sep="")   # Var(ind,dim,cat), where ind gives the position in the final call to KeOps routine, dim is the dimension, and cat the category
    vars = list(x)    # vars lists all actual matrices necessary to evaluate the current formula, here only one.

    # finally we build and return the LazyTensor object
    obj = list(formula = formula, vars=vars, ni=ni, nj=nj)
    class(obj) = "LazyTensor"
    obj
}

unaryop.LazyTensor = function(x,opstr)
{
    if(is.numeric(x))
        x = LazyTensor(x)
    formula = paste(opstr,"(",x$formula,")",sep="")
    obj = list(formula = formula, vars=x$vars, ni=x$ni, nj=x$nj)
    class(obj) = "LazyTensor"
    obj
}

binaryop.LazyTensor = function(x,y,opstr)
{
    if(is.numeric(x))
        x = LazyTensor(x)
    if(is.numeric(y))
        y = LazyTensor(y)
    # update list of variables and update indices in formula
    dec = length(x$vars)
    yform = y$formula
    for(k in 1:length(x$vars))
    {
        str1 = paste("Var(",k-1,sep="")
        str2 = paste("Var(",k-1+dec,sep="")
        yform = gsub(str1, str2, yform, fixed = TRUE)
    }
    formula = paste(x$formula,opstr,yform,sep="")
    vars = c(x$vars,y$vars)
    ni = max(x$ni,y$ni)
    nj = max(x$nj,y$nj)
    obj = list(formula = formula, vars=vars, ni=ni, nj=nj)
    class(obj) = "LazyTensor"
    obj
}

"-.LazyTensor" = function(x,y=NA)
{
    if(length(y)==1 && is.na(y))
        obj = unaryop.LazyTensor(x,"Minus")
    else
        obj = binaryop.LazyTensor(x,y,"-")
}

"^.LazyTensor" = function(x,y)
{
    if(y==2)
        obj = unaryop.LazyTensor(x,"Square")
    else
        stop('not implemented')
}

"*.LazyTensor" = function(x,y)
{
    obj = binaryop.LazyTensor(x,y,"*")
}

"/.LazyTensor" = function(x,y)
{
    obj = binaryop.LazyTensor(x,y,"/")
}

"%*%.default" = .Primitive("%*%") # assign default as current definition

"%*%" = function(x,...)
{ 
    UseMethod("%*%",x)
}

"%*%.LazyTensor" = function(x,y)
{
    if(is.matrix(y))
        y = LazyTensor(y,'j')
    Sum( x*y, index = 'j')
}

Exp <- function(obj,index) 
{
    UseMethod("Exp")
}

Exp.default <- function(obj,index) 
{
    cat("This is a generic function\n")
}

Exp.LazyTensor = function(x)
{
    obj = unaryop.LazyTensor(x,"Exp")
}

reduction.LazyTensor = function(x,opstr,index)
{
    if(index=="i") tag=1 else tag=0
    formula = paste(opstr, "_Reduction(", x$formula, ",", tag, ")", sep = "")
    args = c()
    op = keops_kernel(formula,args)
    res = t(op(x$vars,nx=x$ni,ny=x$nj))
}

Sum <- function(obj,index) 
{
    UseMethod("Sum")
}

Sum.default <- function(obj,index) 
{
    cat("This is a generic function\n")
}

Sum.LazyTensor = function(x,index=NA)
{
    if(is.na(index))
    {
        obj = unaryop.LazyTensor(x,"Sum")
    }
    else
        obj = reduction.LazyTensor(x,"Sum",index)
}



# Basic example

D = 3
M = 100
N = 150
E = 4
x = matrix(runif(M*D),M,D)
y = matrix(runif(N*D),N,D)
b = matrix(runif(N*E),N,E)
s = 0.25

# creating LazyTensor from matrices
x_i  = LazyTensor(x,index='i')  
y_j  = LazyTensor(y,index='j')
b_j  = b

# Symbolic matrix of squared distances: 
SqDist_ij = Sum( (x_i - y_j)^2 )

# Symbolic Gaussian kernel matrix:
K_ij = Exp( - SqDist_ij / (2 * s^2) )

# Genuine matrix: 
v = K_ij %*% b_j

# we compare to standard R computation
SqDist = 0
onesM = matrix(1,1,M)
onesN = matrix(1,1,N)
for(k in 1:D)
    SqDist = SqDist + (x[,k] %*% onesN - t(y[,k] %*% onesM))^2
K = exp(-SqDist/(2*s^2))
v2 = K %*% b   

print(mean(abs(v-v2)))


