#' Defines a new operators
#' @description
#' This function is the core of the KeOps library, it allows you to create 
#' new operators based on kernel operation and matrix reduction discribed as a 
#' mathematic formula.
#' @details
#' The use of the function `keops_kernel` is detailled in the vignettes, 
#' especially how to write formulae, specified input arguments, how to format 
#' data to apply the created operators, etc. Run `browseVignettes("rkeops")` 
#' to access the vignettes.
#' 
#' KeOps operators are defined thanks to formula, i.e. a text string describing 
#' the mathematical operations that you want to apply to your data, and a 
#' list defining the input arguments of your formula.
#' 
#' The function `keops_kernel` compiles and imports a new operator that 
#' implements the formala given in input, it returns a function that can be 
#' used to compute the result of the formula on actual data.
#' 
#' The returned function expects a list of arguments whose order corresponds to 
#' the order given in `args` to `keops_kernel`. We use a list to avoid useless 
#' copies of data.
#' 
#' **Note:** Data are input as a list, because list are references and since 
#' argument passing is done by copy in R, it is better to copy a list of 
#' reference than the actual input data, especially for big matrices.
#' 
#' You should be careful with the input dimension of your data, to correspond 
#' to the input dimension specified in `args`.
#' @author Ghislain Durif
#' @param formula text string, an operator formula (see Details).
#' @param args vector of text string, formula arguments (see Details).
#' @return a function that can be used to compute the value of the formula 
#' on actual data. This function takes as input a list of data corresponding 
#' to the formula arguments and return the computed values (generally a 
#' vector or a matrix depending on the reduction).
#' @importFrom stringr str_length
#' @examples
#' \dontrun{
#' ## Example 1
#' # Define a function that computes for each i the sum over j
#' # of the scalar products of `x_i` and `y_j` (both 3d vectors)
#' F <- keops_kernel(formula = "Sum_Reduction((x|y), 1)",
#'                   args = c("x=Vi(3)", "y=Vj(3)"))
#' ## data
#' nx <- 10
#' ny <- 15
#' # x_i = rows of the matrix X
#' X <- matrix(runif(nx*3), nrow=nx, ncol=3)
#' # y _j = rows of the matrix Y
#' Y <- matrix(runif(ny*3), nrow=ny, ncol=3)
#' ## compute the result
#' res <- F(list(X,Y))
#' 
#' ## Example 2
#' # Define a function that computes the the convolution with a Gauss kernel 
#' # i.e. the sum over j of `e^(lambda*||x_i - y_j||^2) * beta_j` where `x_i`, 
#' # `y_j` and `beta_j` are 3d vectors, and `lambda` is a scalar
#' F = keops_kernel(formula = "Sum_Reduction(Exp(lambda*SqNorm2(x-y))*beta, 1)",
#'                  args = c("x=Vi(3)", "y=Vj(3)", 
#'                           "beta=Vj(3)", "lambda=Pm(1)"))
#' 
#' # data
#' nx <- 10
#' ny <- 15
#' # x_i = rows of the matrix X
#' X <- matrix(runif(nx*3), nrow=nx, ncol=3)
#' # y _j = rows of the matrix Y
#' Y <- matrix(runif(ny*3), nrow=ny, ncol=3)
#' # beta_j = rows of the matrix beta
#' beta <- matrix(runif(ny*3), nrow=ny, ncol=3)
#' # !! important !! y and beta should have the same dimension
#' 
#' # parameter
#' lambda <- 0.25
#' 
#' # compute the result
#' res <- F(list(X, Y, beta, lambda))
#' }
#' @export
keops_kernel <- function(formula, args) {

    # check input
    if(!is.character(formula))
        stop("`formula` input parameter should be a text string")
    if(!(length(args)==0 | (is.vector(args) & is.character(args))))
        stop("`args` input parameter should be a vector of text strings")
    
    # check formula and args formating
    var_aliases <- format_var_aliases(args)
    
    # hash name to compile formula in a shared library file
    dllname <- create_dllname(formula, args)
    dllfilename <- file.path(get_build_dir(), 
                             paste0("librkeops", dllname, .Platform$dynlib.ext))
    
    # compile operator if necessary
    if(!file.exists(dllfilename) | get_rkeops_option("verbosity")) {
        compile_formula(formula, var_aliases$var_aliases, dllname)
    }
    
    # load shared library
    r_genred <- load_dll(path = get_build_dir(),
                         dllname = paste0("librkeops", dllname), 
                         object = "r_genred",
                         genred=TRUE)
    
    # reordering var_aliases (to correspond to operator input)
    var_aliases <- lapply(var_aliases, function(elem)
        if(length(elem)>1)
            return(elem[order(var_aliases$var_pos)])
        else
            return(elem))
    
    
    # return function calling the corresponding compile operator
    function(input=NULL) {
        ## !! important !!
        # input: should be a list, because list are references and since 
        #   argument passing is done by copy in R, it is better to copy a list 
        #   of reference than the actual input data, especially for big 
        #   matrices.
        # if NULL or missing, env (context) is returned
        
        ## storing some context
        env <- list(formula=formula,
                    args=args,
                    var_aliases=var_aliases)
        if(missing(input) | is.null(input))
            return(env)
        
        ## reorder input if names are supplied (if not list order is used)
        # check that all input args are named
        if(sum(str_length(names(input)) > 0) == length(input)) {
            # expected order
            expected_order <- env$var_aliases$var_name
            # check if names are consistant
            if(all(names(input) %in% expected_order))
                if(any(names(input) != expected_order))
                    input <- input[expected_order]
        }
        
        ## transform scalar to matrix (generally parameters)
        check_scalar <- sapply(1:length(input), 
                               function(ind) return(is.null(dim(input[[ind]]))))
        if(any(check_scalar)) {
            tmp_names <- names(input)
            input[check_scalar] <- lapply(which(check_scalar),
                                          function(ind) 
                                              return(as.matrix(input[[ind]])))
            names(input) <- tmp_names
        }

        ## transpose input if necessary
        # check if necessary
        check_input_dim <- sapply(1:length(input),
            function(ind) {
                input_dim <- nrow(input[[ind]])
                expected_dim <- env$var_aliases$var_dim[ind]
                return(input_dim != expected_dim)
            })
        # transpose if necessary
        if(any(check_input_dim)) {
            tmp_names <- names(input)
            input[check_input_dim] <- lapply(which(check_input_dim),
                    function(ind) {
                        input_dim <- nrow(input[[ind]])
                        expected_dim <- env$var_aliases$var_dim[ind]
                        return(t(input[[ind]]))
                    })
            names(input) <- tmp_names
        }

        ## range i
        index_i <- which(env$var_aliases$var_type == "Vi")
        nxs <- sapply(input[index_i], ncol)
        nx <- nxs[1]
        if(mean(nxs) != nx) stop("Range of index i is different among all Vi's variables")
        ## range j
        index_j <- which(env$var_aliases$var_type == "Vj")
        nys <- sapply(input[index_j], ncol)
        ny <- nys[1]
        if(mean(nys) != ny) stop("Range of index j is different among all Vj's variables")
        
        ## run
        param <- c(get_rkeops_options("runtime"),
                   list(nx=nx, ny=ny))
        return(r_genred(input, param))
    }
}