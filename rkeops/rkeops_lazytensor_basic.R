library(rkeops)


# Minimal LazyTensor implementation

LazyTensor = function(x,index)
{
    ni = 0
    nj = 0
    if(is.matrix(x))
    {
        # x is matrix, treated as indexed variable, so index must be "i" or "j"
        d = nrow(x)
        if(index=='i')
        {
            cat=0 
            ni = ncol(x)
        }
        else
        {
            cat = 1
            nj = ncol(x)
        }
    }
    else
    {
        # assume x is numeric vector, treated as parameter, then converted to matrix
        d = length(x)
        cat = 2
        x = matrix(x)
    }
    formula = paste('Var(0,',d,',',cat,')', sep="")
    vars = list(x)
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
    if(index=="i") tag=0 else tag=1 
    formula = paste(opstr, "_Reduction(", x$formula, ",", tag, ")", sep = "")
    args = c()
    op = keops_kernel(formula,args)
    param <- list(tagCpuGpu=0, tag1D2D=0, tagHostDevice=0, Device_Id=0, nx=x$ni, ny=x$nj)
    res = op(x$vars,param)
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
x = matrix(runif(M,D), nrow=D)
y = matrix(runif(N,D), nrow=D)
b = matrix(runif(N,E), nrow=E)
s = 0.25

# creating LazyTensor objects from matrices
x_i  = LazyTensor(x,index='i')  
x_j  = LazyTensor(x,index='j')
b_j  = LazyTensor(b,index='j')

# Symbolic matrix of squared distances: 
D_ij = Sum( (x_i - x_j)^2 )

# Symbolic Gaussian kernel matrix:
K_ij = Exp( - D_ij / (2 * s^2) )

# Genuine matrix: 
v = Sum(K_ij * b_j, index='j')  
