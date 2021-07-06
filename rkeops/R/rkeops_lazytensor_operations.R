library(rkeops)
library(stringr)


set_rkeops_option("tagCpuGpu", 0)
set_rkeops_option("precision", "double")

# TODO redo doc

#' Build and return a LazyTensor object
#' @description
#' LazyTensors objects are wrappers around R matrices or vectors that are used to create
#' symbolic formulas for the KeOps reduction operations.
#' @details
#' The use of the function `LazyTensor` is detailed in the vignettes. 
#' Run `browseVignettes("rkeops")` to access the vignettes.
#' @author Ghislain Durif
#' @param x A matrix or a vector of numeric values, or a scalar value
#' @param index A text string that should be either `"i"` or `"j"`, or an `NA` value (the default),
#' to specify whether if the `x` variable is indexed by i, by j, or is a fixed parameter across indices.
#' If `x` is a matrix, `index` must be `"i"` or `"j"`.
#' @return An object of class "LazyTensor", which is a list with the following elements:
#' @return
#' \itemize{
#'     \item{`formula`:}{ A string defining the mathematical operation to be computed by the KeOps routine, of the form Var(ind,dim,cat), where :
#'     \itemize{
#'         \item{`ind`:}{ gives the position in the final call to KeOps routine}
#'         \item{`dim`:}{ is the dimension of the input}
#'         \item{`cat`:}{ the KeOps "category": 0 if the input is an R matrix indexed by "i", 1 if the input is an R matrix indexed by "j", or 2 if the input is a parameter vector or scalar, without any attached index}
#'     }}
#'     \item{`vars`:}{ a list of R matrices which will be the inputs of the KeOps routine}
#'     \item{`ni`:}{ the number of rows of the input if it is an "i" indexed variable}
#'     \item{`nj`:}{ the number of rows of the input if it is a "j" indexed variable}
#' }
#' @examples
#' \dontrun{
#' # TODO change set_rkeops_options() below ?
#' set_rkeops_options()
#' 
#' # Data
#' # TODO increase `nx` and `ny`
#' nx <- 100
#' ny <- 150
#' x <- matrix(runif(nx * 3), nrow = nx, ncol = 3) # arbitrary R matrix representing 100 data points in R^3
#' y <- matrix(runif(ny * 3), nrow = ny, ncol = 3) # arbitrary R matrix representing 150 data points in R^3
#' s <- 0.1                                        # scale parameter
#' 
#' # Turn our Tensors into KeOps symbolic variables:
#' x_i <- LazyTensor(x, "i")   # symbolic object representing an arbitrary row of x, indexed by the letter "i"
#' y_j <- LazyTensor(y, "j")   # symbolic object representing an arbitrary row of y, indexed by the letter "j"
#' 
#' # Perform large-scale computations, without memory overflows:
#' D_ij <- Sum((x_i - y_j)^2)    # symbolic matrix of pairwise squared distances, with 100 rows and 150 columns
#' K_ij <- Exp(- D_ij / s^2)     # symbolic matrix, 100 rows and 150 columns
#' # TODO fix `reduction.LazyTensor`
#' res <- Sum(K_ij, index = "i") # actual R matrix (in fact a row vector of length 150 here)
#'                               # containing the column sums of K_ij
#'                               # (i.e. the sums over the "i" index, for each "j" index)
#'
#' }
#' @export
LazyTensor <- function(x, index = NA)
{
    # init
    d = NULL
    cat = NULL
    
    if(is.character(x))
        stop("`x` input argument should be a matrix, a vector or a scalar.")
    if(is.matrix(x) && is.na(index))
        stop("missing `index` argument")
    

    # 1) input is a matrix, treated as indexed variable, so index must be "i" or "j"
    if(is.matrix(x))
    {
        d <- ncol(x)
        if(index=='i')
        {
            # cat = 0
            cat = "Vi"
        }
        else
        {
            # cat = 1
            cat = "Vj"
        }
    }
    # 2) else we assume x is a numeric vector, treated as parameter, then converted to matrix
    else
    {
        d <- length(x)
        # cat <- 2
        cat <- "Pm"
    }

    # Now we define "formula", a string specifying the variable for KeOps C++ codes.
    var_name = "var0"
    formula = var_name
    # formula <- paste('Var(0,', d, ',', cat, ')', sep = "")  # Var(ind,dim,cat), where :
    #                                                         # ind gives the position in the final call to KeOps routine,
    #                                                         # dim is the dimension
    #                                                         # cat the category
    vars <- list(x)  # vars lists all actual matrices necessary to evaluate the current formula, here only one.
    args = str_c(var_name, "=", cat, "(", d, ")")

    # finally we build and return the LazyTensor object
    obj <- list(formula = formula, args = args, vars = vars)
    class(obj) <- "LazyTensor"
    return(obj)
}

# TODO error print when matrix in input

#' @export
unaryop.LazyTensor <- function(x,opstr)
{
    if(is.matrix(x))
        stop("`x` input argument should be a LazyTensor, a vector or a scalar.")
    
    if(is.numeric(x))
        x <- LazyTensor(x)
    formula <- paste(opstr, "(", x$formula, ")", sep="")
    obj <- list(formula = formula, args=x$args, vars=x$vars)
    class(obj) <- "LazyTensor"
    return(obj)
}


#binaryop.LazyTensor = function(x,y,opstr)
#{
#    if(is.numeric(x))
#        x = LazyTensor(x)
#    if(is.numeric(y))
#        y = LazyTensor(y)
#    # update list of variables and update indices in formula
#    dec = length(x$vars)
#    yform = y$formula
#    for(k in 1:length(x$vars))
#    {
#        str1 = paste("Var(",k-1,sep="")
#        str2 = paste("Var(",k-1+dec,sep="")
#        yform = gsub(str1, str2, yform, fixed = TRUE)
#    }
#    formula = paste(x$formula,opstr,yform,sep="")
#    vars = c(x$vars,y$vars)
#    ni = max(x$ni,y$ni)
#    nj = max(x$nj,y$nj)
#    obj = list(formula = formula, vars=vars, ni=ni, nj=nj)
#    class(obj) = "LazyTensor"
#    obj
#}


# opstr : string, the operation
# is_operator : boolean, TRUE if the operation 'opstr' is an operator like "+" or "-"
# TODO : pb if y is a vector or a matrix etc ?

#' @export
binaryop.LazyTensor <- function(x, y, opstr, is_operator=FALSE)
{
    if(is.numeric(x))
        x <- LazyTensor(x)
    
    # if y is a scalar and the operation is a specific operation
    # for instance we want : Pow(var0,2)
    op_specific <- list("Pow") 
    if(is.element(opstr, op_specific) && class(y) != "LazyTensor"){
        if(is_operator)
            formula <- paste(x$formula, opstr, y, sep="")
        # case when the operation is not an operator
        else 
            formula <- paste(opstr, "(", x$formula, ",", y, ")", sep="")
        vars <- c(x$vars, y) # TODO check if we need y$vars instead of y
        args <- c(x$args, y) # TODO check if we need y$args instead of y
    }
    # case with no specific operation 
    else{
        if(is.numeric(y))
            y <- LazyTensor(y)
        
        dec <- length(x$vars)
        yform <- y$formula
        # update list of variables and update indices in formula
        for(k in 1:length(y$vars))
        {
            str1 <- paste("var", k - 1, sep="")
            str2 <- paste("var", k - 1 + dec, sep="")
            yform <- gsub(str1, str2, yform, fixed = TRUE)
            y$args[k] <- gsub(str1, str2, y$args[k], fixed = TRUE)
        }
        # special formula for operator
        if(is_operator)
            formula <- paste(x$formula, opstr, yform, sep="")
        else
            formula <- paste(opstr, "(", x$formula, ",", yform, ")", sep="")
        vars <- c(x$vars,y$vars)
        args <- c(x$args,y$args)
    }
    
    obj <- list(formula = formula, args=args, vars=vars)
    class(obj) <- "LazyTensor"
    return(obj)
}


"-.LazyTensor" <- function(x, y=NA)
{
    if(length(y)==1 && is.na(y))
        obj <- unaryop.LazyTensor(x, "Minus")
    else
        obj <- binaryop.LazyTensor(x, y, "-", is_operator = TRUE)
}

"^.default" <- .Primitive("^") # assign default as current definition

"^" <- function(x, ...)
{ 
    UseMethod("^", x)
}

"^.LazyTensor" <- function(x, y)
{
    if(is.numeric(y) && (as.integer(y)-y) == 0){
        if(y == 2)
            obj <- unaryop.LazyTensor(x, "Square")
        else
            obj <- binaryop.LazyTensor(x, y, "Pow")
    }
    
    else if(is.numeric(y) && y == 0.5)
        obj <- unaryop.LazyTensor(x, "Sqrt") # element-wise square root
    
    else if(is.numeric(y) && y == (-0.5))
        obj <- unaryop.LazyTensor(x, "Rsqrt") # element-wise inverse square root
    
    else
        obj <- binaryop.LazyTensor(x, y, "Powf") # power operation
    
    return(obj)
}

Square <- function(x){
    obj <- unaryop.LazyTensor(x, "Square")
}

Sqrt <- function(x){
    obj <- unaryop.LazyTensor(x, "Sqrt")
}

# multiplication
"*.default" <- .Primitive("*") # assign default as current definition

"*" <- function(x, ...)
{ 
    UseMethod("*", x)
}

"*.LazyTensor" <- function(x, y)
{
    obj <- binaryop.LazyTensor(x, y, "*", is_operator = TRUE)
}

# division
"/.default" <- .Primitive("/") # assign default as current definition

"/" <- function(x, ...)
{ 
    UseMethod("/", x)
}

"/.LazyTensor" <- function(x, y)
{
    obj <- binaryop.LazyTensor(x, y, "/", is_operator = TRUE)
}

"|.LazyTensor" <- function(x,y)
{
    obj <- binaryop.LazyTensor(x, y, "|", is_operator = TRUE)
    obj$formula <- paste("(", obj$formula, ")", sep = "")
    obj
}

"%*%.default" <- .Primitive("%*%") # assign default as current definition

"%*%" <- function(x, ...)
{ 
    UseMethod("%*%", x)
}

"%*%.LazyTensor" <- function(x, y)
{
    if(is.matrix(y))
        y <- LazyTensor(y,'j')
    Sum( x*y, index = 'j')
}

Exp <- function(x) # remove the `index` argument
{
    UseMethod("Exp")
}

Exp.default <- function(x) 
{
    cat("This is a generic function\n")
}

Exp.LazyTensor <- function(x)
{
    obj <- unaryop.LazyTensor(x, "Exp")
}

Log <- function(x) # remove the `index` argument
{
    UseMethod("Log")
}

Log.default <- function(x) 
{
    cat("This is a generic function\n")
}

Log.LazyTensor <- function(x)
{
    obj <- unaryop.LazyTensor(x, "Log")
}

# TODO : this function doesn't work 
reduction.LazyTensor <- function(x,opstr,index)
{
    if(index=="i") tag<-1 else tag<-0
    formula <- paste(opstr, "_Reduction(", x$formula, ",", tag, ")", sep = "")
    args = x$args
    op <- keops_kernel(formula,args)
    res <- op(x$vars)
    return(res)
}

Sum <- function(obj, index) 
{
    UseMethod("Sum")
}

Sum.default <- function(obj, index) 
{
    cat("This is a generic function\n")
}

Sum.LazyTensor <- function(x, index=NA)
{
    if(is.na(index))
    {
        obj <- unaryop.LazyTensor(x,"Sum")
    }
    else
        obj <- reduction.LazyTensor(x,"Sum",index)
}

# element-wise inverse 1/x
Inv <- function(x){
    obj <- unaryop.LazyTensor(x, "Inv")
}

Sin <- function(x){
    obj <- unaryop.LazyTensor(x, "Sin")
}

Asin <- function(x){
    obj <- unaryop.LazyTensor(x, "Asin")
}

Cos <- function(x){
    obj <- unaryop.LazyTensor(x, "Cos")
}

Acos <- function(x){
    obj <- unaryop.LazyTensor(x, "Acos")
}

Atan <- function(x){
    obj <- unaryop.LazyTensor(x, "Atan")
}

SqNorm2 <- function(x){
    obj <- unaryop.LazyTensor(x, "SqNorm2")
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
# equivalent
# v = "%*%.LazyTensor"(K_ij, b_j)

s2 = (2 * s^2)
# equivalent
op <- keops_kernel(
    formula = "Sum_Reduction(Exp(Minus(Sum(Square(x-y)))/s)*b,0)",
    args = c("x=Vi(3)", "y=Vj(3)", "s=Pm(1)", "b=Vj(4)")
)


v2 <- op(list(x, y, s2, b))

sum((v2-v)^2)



# we compare to standard R computation
SqDist = 0
onesM = matrix(1,1,M)
onesN = matrix(1,1,N)
for(k in 1:D)
    SqDist = SqDist + (x[,k] %*% onesN - t(y[,k] %*% onesM))^2
K = exp(-SqDist/(2*s^2))
v2 = K %*% b

print(mean(abs(v-v2)))


