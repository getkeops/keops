#' Compile a formula defining a new operator
#' @keywords internal
#' @description
#' User-defined operators associated to a formula are compiled in shared object 
#' files with this function.
#' @details
#' The function `compile_formula` sets up the working directory and the 
#' environment to call the function [rkeops::compile_code()] for the 
#' cmake compilation.
#' 
#' Afterwards, it also verify that the compilation went smoothly.
#' @param formula text string, formula defining the new operator.
#' @param var_aliases text string, formated formula input arguments returned by 
#' [rkeops::format_var_aliases()] (specifically `$var_aliases`).
#' @param dllname text string, the name associated to the related shared object 
#' file.
#' @return 0 if no problem
#' @author Ghislain Durif
#' @seealso [rkeops::keops_kernel()], [rkeops::compile_code()]
#' @export
compile_formula <- function(formula, var_aliases, dllname) {

    ## cmake src dir
    cmake_dir <- dirname(get_rkeops_option("src_dir"))
    
    ## get user current working directory
    current_directory <- getwd()
    
    ## generate binder
    tmp <- compileAttributes(pkgdir = file.path(get_src_dir(),
                                                "binder"))
    if(length(tmp) == 0) {
        stop("Issue when generating R/C++ interface")
    }
    
    ## move to build directory
    tmp_build_dir <- file.path(get_rkeops_option("build_dir"), dllname)
    if(!dir.exists(tmp_build_dir)) dir.create(tmp_build_dir)
    setwd(tmp_build_dir)
    on.exit(setwd(current_directory))
    
    ## compiling (call to cmake)
    return_status <- compile_code(formula, var_aliases, dllname, cmake_dir)
    
    ## move back to user working directory
    setwd(current_directory)
    
    ## check compiling status
    if(return_status != 0)
        stop("Error during cmake call.")
    
    ## check compilation
    test_binder <- tryCatch(load_dll(path = tmp_build_dir,
                                     dllname = paste0("librkeops", dllname), 
                                     object = "test_binder"),
                            error = function(e) return(NULL))
    if(is.null(test_binder) | 
       !tryCatch(test_binder(), error = function(e) return(FALSE)))
        stop("Issue with compilation.")
    
    ## cleaning tmp build dir
    so_file_list <- file.path(tmp_build_dir, 
                              paste0(c("librkeops", "lib", ""), 
                                     dllname, .Platform$dynlib.ext))
    file.copy(from=so_file_list, to=get_rkeops_option("build_dir"), 
              overwrite = TRUE, recursive = FALSE, 
              copy.mode = TRUE)
    unlink(tmp_build_dir, recursive = TRUE)
    ## output
    return(0)
}
