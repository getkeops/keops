library(rkeops)
library(stringr)
library(data.table)

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
#' @param index A text string that should be either **i** or **j**, or an **NA** value (the default),
#' to specify whether if the **x** variable is indexed by **i**, by **j**, or is a fixed parameter across indices.
#' If **x** is a matrix, **index** must be **i** or **j**.
#' @return An object of class "LazyTensor", which is a list with the following elements:
#' @return
#' \itemize{
#'     \item{**formula**:}{ A string defining the mathematical operation to be computed by the KeOps routine}
#'     \item{**args**:}{ A vector of arguments containing a unique identifier associated to type of the argument :
#'     \itemize{
#'         \item{**Vi(n)**:}{ vector indexed by **i** of dim **n**}
#'         \item{**Vj(n)**:}{ vector indexed by **j** of dim **n**}
#'         \item{**Pm(n)**:}{ fixed parameter of dim **n**}
#'     }}
#'     \item{**vars**:}{ A list of R matrices which will be the inputs of the KeOps routine}
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
#' D_ij <- sum((x_i - y_j)^2)    # symbolic matrix of pairwise squared distances, with 100 rows and 150 columns
#' K_ij <- exp(- D_ij / s^2)     # symbolic matrix, 100 rows and 150 columns
#' res <- sum(K_ij, index = "i") # actual R matrix (in fact a row vector of length 150 here)
#'                               # containing the column sums of K_ij
#'                               # (i.e. the sums over the "i" index, for each "j" index)
#'
#' }
#' @export
LazyTensor <- function(x, index = NA)
{
    # init
    d <- NULL
    cat <- NULL
    
    if(is.character(x))
        stop("`x` input argument should be a matrix, a vector or a scalar.")
    if(is.matrix(x) && is.na(index))
        stop("missing `index` argument")
    

    # 1) input is a matrix, treated as indexed variable, so index must be "i" or "j"
    if(is.matrix(x))
    {
        d <- ncol(x)
        if(index == "i")
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
    #var_name = "var0"
    #var_name = address(x)
    #var_name = paste("A", address(x), sep="")
    var_name <- paste("A", address(x), index, sep = "") 
    formula <- var_name
    # formula <- paste('Var(0,', d, ',', cat, ')', sep = "")  # Var(ind,dim,cat), where :
    #                                                         # ind gives the position in the final call to KeOps routine,
    #                                                         # dim is the dimension
    #                                                         # cat the category
    vars <- list(x)  # vars lists all actual matrices necessary to evaluate the current formula, here only one.
    #args = str_c(var_name, "=", cat, "(0,", d, ")")
    args <- str_c(var_name, "=", cat, "(", d, ")")
    # finally we build and return the LazyTensor object
    obj <- list(formula = formula, args = args, vars = vars)
    class(obj) <- "LazyTensor"
    return(obj)
}


#' Build a unary operation
#' @description
#' Symbolically applies **opstr** operation to **x**.
#' @author Ghislain Durif
#' @param x A LazyTensor, a vector of numeric values, or a scalar value.
#' @param opstr A text string corresponding to an operation.
#' @return An object of class "LazyTensor".
#' @examples
#' \dontrun{
#' }
#' @export
unaryop.LazyTensor <- function(x,opstr) {
    if(is.matrix(x))
        stop("`x` input argument should be a LazyTensor, a vector or a scalar.")
    
    if(is.numeric(x))
        x <- LazyTensor(x)
    formula <- paste(opstr, "(", x$formula, ")", sep="")
    obj <- list(formula = formula, args = x$args, vars = x$vars)
    class(obj) <- "LazyTensor"
    return(obj)
}

#' Build a binary operation
#' @description
#' Symbolically applies **opstr** operation to **x** and **y**.
#' @author Ghislain Durif
#' @param x A LazyTensor, a vector of numeric values, or a scalar value.
#' @param y A LazyTensor, a vector of numeric values, or a scalar value.
#' @param opstr A text string corresponding to an operation.
#' @param is_operator A boolean used to specify if **opstr** is an operator like ``+``
#' , ``-`` or a "genuine" function.
#' @return An object of class "LazyTensor".
#' @examples
#' \dontrun{
#' }
#' @export
binaryop.LazyTensor <- function(x, y, opstr, is_operator=FALSE) {
    if(is.matrix(x))
        stop("`x` and  input argument should be a LazyTensor, a vector or a scalar.")
    
    if(is.matrix(y))
        stop("`y` and  input argument should be a LazyTensor, a vector or a scalar.")
    
    if(is.numeric(x))
        x <- LazyTensor(x)
    
    # if y is a scalar and the operation is a specific operation
    # for instance we want : Pow(x,2)
    op_specific <- list("Pow", "Round") 
    condition1 <- is.element(opstr, op_specific) && is.numeric(y) && (as.integer(y) - y) == 0
    condition2 <- opstr == "Round" && is.numeric(y) && length(y) == 1
    if(condition1 || condition2){
        if(is_operator)
            formula <- paste(x$formula, opstr, y, sep = "")
        # case when the operation is not an operator
        else 
            formula <- paste(opstr, "(", x$formula, ",", y, ")", sep="")
        vars <- c(x$vars)
        args <- c(x$args)
    }
    # case with no specific operation 
    else{
        if(is.numeric(y))
            y <- LazyTensor(y)
        
        dec <- length(x$vars)
        yform <- y$formula
        # update list of variables and update indices in formula
        #for(k in 1:length(y$vars))
        #{
            #add = substr(y$args[k],1,14)
            #begin <- substr(y$args[k], 1, 18)
            #end <- substr(y$args[k], 19, 22)
            #str1  <- paste(begin, k - 1, ",", end, sep="")
            #str2 <- paste(begin, k - 1 + dec, ",", end, sep="")
            #str1 <- paste("(", k - 1, ",", sep="")
            #str2 <- paste("(", k - 1 + dec, ",", sep="")
            #str3 <- paste(add, "_", k - 1, sep="")
            #str4 <- paste(add, "_", k - 1 + dec, sep="")
            #yform <- gsub(str1, str2, yform, fixed = TRUE)
            #y$args[k] <- gsub(str1, str2, y$args[k], fixed = TRUE) # to improve
            #yform <- gsub(str3, str4, yform, fixed = TRUE)
        #}
        # special formula for operator
        if(is_operator)
            formula <- paste(x$formula, opstr, yform, sep="")
        else
            formula <- paste(opstr, "(", x$formula, ",", yform, ")", sep="")
        vars <- c(x$vars,y$vars)
        vars[!duplicated(names(vars))]
        args <- unique(c(x$args,y$args))
    }
    
    obj <- list(formula = formula, args=args, vars=vars)
    class(obj) <- "LazyTensor"
    return(obj)
}


# soustraction
"-.default" <- .Primitive("-") # assign default as current definition

"-" <- function(x, y = NA) { 
    if(class(x)[1] != "LazyTensor")
        UseMethod("-", y)
    else
        UseMethod("-", x)
}

"-.LazyTensor" <- function(x, y = NA) {
    if((length(y) == 1) && is.na(y))
        obj <- unaryop.LazyTensor(x, "Minus")
    else
        obj <- binaryop.LazyTensor(x, y, "-", is_operator = TRUE)
}

"^.default" <- .Primitive("^") # assign default as current definition

"^" <- function(x, y) { 
    if(class(x)[1] != "LazyTensor")
        UseMethod("^", y)
    else
        UseMethod("^", x)
}

"^.LazyTensor" <- function(x, y) {   
    if(is.numeric(y) && length(y) == 1){
        if((as.integer(y) - y) == 0){
            if(y == 2)
                obj <- unaryop.LazyTensor(x, "Square")
            else
                obj <- binaryop.LazyTensor(x, y, "Pow")
        }
        
        else if(y == 0.5)
            obj <- unaryop.LazyTensor(x, "Sqrt") # element-wise square root
        
        else if(y == (-0.5))
            obj <- unaryop.LazyTensor(x, "Rsqrt") # element-wise inverse square root
        
        # check if Powf with y a float number has to be like Powf(var1,var2) or Powf(var,y) (Powf(var, 0.5))
        else
            obj <- binaryop.LazyTensor(x, y, "Powf") # power operation
        
        }
    
    else
        obj <- binaryop.LazyTensor(x, y, "Powf") # power operation
    
    return(obj)
}


# square
square <- function(x) {
    obj <- unaryop.LazyTensor(x, "Square")
}


# square root
sqrt.default <- .Primitive("sqrt") # assign default as current definition

sqrt <- function(x) { 
    UseMethod("sqrt", x)
}

sqrt.LazyTensor <- function(x) {
    obj <- unaryop.LazyTensor(x, "Sqrt")
}

# Rsqrt
rsqrt <- function(x) {
    obj <- unaryop.LazyTensor(x, "Rsqrt")
}

# addition
# Broadcasted addition operator, a binary operation.
# x + y returns a class `LazyTensor` that encodes,
# symbolically, the addition of ``x`` and ``y``.
"+.default" <- .Primitive("+") # assign default as current definition

"+" <- function(x, y) { 
    if(class(x)[1] != "LazyTensor")
        UseMethod("+", y)
    else
        UseMethod("+", x)
}

"+.LazyTensor" <- function(x, y) {
    obj <- binaryop.LazyTensor(x, y, "+", is_operator = TRUE)
}


# multiplication
"*.default" <- .Primitive("*") # assign default as current definition

"*" <- function(x, y) { 
    if(class(x)[1] != "LazyTensor")
        UseMethod("*", y)
    else
        UseMethod("*", x)
}

"*.LazyTensor" <- function(x, y) {
    obj <- binaryop.LazyTensor(x, y, "*", is_operator = TRUE)
}

# division
"/.default" <- .Primitive("/")

"/" <- function(x, y) { 
    if(class(x)[1] != "LazyTensor")
        UseMethod("/", y)
    else
        UseMethod("/", x)
}

"/.LazyTensor" <- function(x, y) {
    obj <- binaryop.LazyTensor(x, y, "/", is_operator = TRUE)
}

# scalar product
"|.default" <- .Primitive("|")

"|" <- function(x, y) { 
    if(class(x)[1] != "LazyTensor")
        UseMethod("|", y)
    else
        UseMethod("|", x)
}
"|.LazyTensor" <- function(x, y) {
    obj <- binaryop.LazyTensor(x, y, "|", is_operator = TRUE)
    obj$formula <- paste("(", obj$formula, ")", sep = "")
    obj
}


"%*%.default" <- .Primitive("%*%") # assign default as current definition

"%*%" <- function(x, y) { 
    if(class(x)[1] != "LazyTensor")
        UseMethod("%*%", y)
    else
        UseMethod("%*%", x)
}

"%*%.LazyTensor" <- function(x, y) {
    if(is.matrix(y))
        y <- LazyTensor(y, "j")
    sum(x * y, index = "j")
}

# exponential
exp.default <- .Primitive("exp")

exp <- function(x) {
    UseMethod("exp")
}

exp.LazyTensor <- function(x) {
    obj <- unaryop.LazyTensor(x, "Exp")
}


# logarithm
log.default <- .Primitive("log")

log <- function(x) 
{
    UseMethod("log")
}

log.LazyTensor <- function(x)
{
    obj <- unaryop.LazyTensor(x, "Log")
}

# element-wise inverse 1/x
# TODO doc explain that inv(scalar) = 1/scalar and inv(matrix) = matrix^{-1} and 1/vector = element-wise inverse
inv.default <- function(x) {
    if(is.matrix(x))
        res <- solve(x)
    else
        res <- 1 / x
    return(res)
}

inv <- function(x) {
    UseMethod("inv")
}

inv.LazyTensor <- function(x) {
    obj <- unaryop.LazyTensor(x, "Inv")
}


# cosinus
cos.default <- .Primitive("cos")

cos <- function(x) {
    UseMethod("cos")
}

cos.LazyTensor <- function(x) {
    obj <- unaryop.LazyTensor(x, "Cos")
}

# sinus
sin.default <- .Primitive("sin")

sin <- function(x) 
{
    UseMethod("sin")
}

sin.LazyTensor  <- function(x){
    obj <- unaryop.LazyTensor(x, "Sin")
}

# arccos
acos.default <- .Primitive("acos")

acos <- function(x) {
    UseMethod("acos")
}
acos.LazyTensor <- function(x) {
    obj <- unaryop.LazyTensor(x, "Acos")
}

# arcsinus
asin.default <- .Primitive("asin")

asin <- function(x) {
    UseMethod("asin")
}

asin.LazyTensor <- function(x) {
    obj <- unaryop.LazyTensor(x, "Asin")
}

# arctan
atan.default <- .Primitive("atan")

atan <- function(x) {
    UseMethod("atan")
}

atan.LazyTensor <- function(x) {
    obj <- unaryop.LazyTensor(x, "Atan")
}

# element-wise 2-argument arc-tangent function
atan2.default <- function(x, y) {
    .Internal(atan2(x, y))
}

atan2 <- function(x, y) {
    if(class(x)[1] != "LazyTensor")
        UseMethod("atan2", y)
    else
        UseMethod("atan2", x)
}

atan2.LazyTensor <- function(x, y) {
    obj <- binaryop.LazyTensor(x, y, "Atan2")
}


# absolute value
abs.default <- .Primitive("abs")

abs <- function(x) {
    UseMethod("abs")
}

abs.LazyTensor <- function(x) {
    obj <- unaryop.LazyTensor(x, "Abs")
}


# sign function
sign.default <- .Primitive("sign")

sign <- function(x) {
    UseMethod("sign")
}

sign.LazyTensor <- function(x) {
    obj <- unaryop.LazyTensor(x, "Sign")
}


# Round function
round.default <- .Primitive("round")

round <- function(x, ...) {
    UseMethod("round", x)
}

round.LazyTensor <- function(x, y) {
    obj <- binaryop.LazyTensor(x, y, "Round")
}


# Préciser que si on a plusieurs scalaires, on peut faire e.g. min(3, 4, 11)
# qui renvoie 11 mais pour les LazyTensor c'est juste min(x_i) qui renvoie
# l'élément minimal de x_i
# Min function
min.default <- .Primitive("min")

min <- function(x, ...) {
    UseMethod("min")
}

min.LazyTensor <- function(x) {
    obj <- unaryop.LazyTensor(x, "Min")
}

# Préciser que si on a plusieurs scalaires, on peut faire e.g. max(3, 4, 11)
# qui renvoie 11 mais pour les LazyTensor c'est juste max(x_i) qui renvoie
# l'élément maximal de x_i
# Max function
max.default <- .Primitive("max")

max <- function(x, ...) {
    UseMethod("max", x)
}

max.LazyTensor <- function(x) {
    obj <- unaryop.LazyTensor(x, "Max")
}


# xlogx function
# TODO value 0 at 0 c'est sûr ??
xlogx.default <- function(x) {
    if(x == 0)
        res <- 0
    else
        res <- x * log(x)
    return(res)
}

xlogx <- function(x) {
    UseMethod("xlogx", x)
}

xlogx.LazyTensor <- function(x){
    obj <- unaryop.LazyTensor(x, "XLogX")
}


# sinxdivx function
# TODO value 1 at 0 ?
sinxdivx.default <- function(x) {
    if(x == 0)
        res <- 1
    else
        res <- sin(x) / x
    return(res)
}

sinxdivx <- function(x) {
    UseMethod("sinxdivx", x)
}

sinxdivx.LazyTensor <- function(x){
    obj <- unaryop.LazyTensor(x, "SinXDivX")
}


# Reduction
reduction.LazyTensor <- function(x, opstr, index) {
    if(index == "i") 
        tag <- 1
    else 
        tag <- 0
    formula <- paste(opstr, "_Reduction(", x$formula, ",", tag, ")", sep = "")
    args <- x$args
    op <- keops_kernel(formula, args)
    res <- op(x$vars)
    return(res)
}

# sum function
sum.default <- .Primitive("sum")

sum <- function(obj, index) {
    UseMethod("sum")
}

sum.LazyTensor <- function(x, index = NA) {
    if(is.na(index))
        obj <- unaryop.LazyTensor(x, "Sum")
    else if(is.character(index))
        obj <- reduction.LazyTensor(x, "Sum", index)
    else
        stop("`index` input argument should be a character `i`, `j` or NA.")
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
x_i = LazyTensor(x, index = 'i')
y_j = LazyTensor(y, index = 'j')
b_j = b

# Symbolic matrix of squared distances:
SqDist_ij = sum( (x_i - y_j)^2 )

# Symbolic Gaussian kernel matrix:
K_ij = exp( - SqDist_ij / (2 * s^2) )

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


