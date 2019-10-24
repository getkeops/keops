#' Defines a new operators
#' @description
#' FIXME
#' @details
#' FIXME
#' See <http://www.kernel-operations.io/keops/api/math-operations.html>
#' @author Ghislain Durif
#' @param formula text string, an operator formula.
#' @param args vector of text string, operator parameters.
#' @return
#' FIXME
#' @importFrom stringr str_length
#' @examples
#' \dontrun{
#' # Define and test a function that computes for each i the sum over j
#' # of the square of the scalar products of `x_i` and `y_j` (both 3d vectors)
#' F <- keops_kernel(formula = "Sum_Reduction(Square((x,y)),0)",
#'                   args = c("x=Vi(3)", "y=Vj(3)"))
#' x <- rnorm(3,2000)
#' y <- rnorm(3,5000)
#' res <- F(x,y)
#' 
#' # Define and test the convolution with a Gauss kernel i.e. the sum 
#' # over j of `e^(lambda*||x_i - y_j||^2) * beta_j` where `x_i`, `y_j` 
#' # and `beta_j` are 3d vectors
#' F = keops_kernel(formula = "Sum_Reduction(Exp(lambda*SqNorm2(x-y))*beta,0)",
#'                  args = c("x=Vi(3)", "y=Vj(3)", 
#'                           "beta=Vj(3)", "lambda=Pm(1)"))
#' x <- rnorm(3,2000)
#' y <- rnorm(3,5000)
#' beta <- rnorm(3,5000)
#' lambda <- 0.25
#' res <- F(x, y, beta, lambda)
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
    function(input) {
        ## storing some context
        env <- list(formula=formula,
                    args=args,
                    var_aliases=var_aliases)
        
        ## reorder input if names are supplied
        # check that all input args are named
        if(sum(str_length(names(input)) > 0) == length(input)) {
            # expected order
            expected_order <- env$var_aliases$var_name
            # check if names are consistant
            if(all(names(input) %in% expected_order))
                if(any(names(input) != expected_order))
                    input <- input[expected_order]
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