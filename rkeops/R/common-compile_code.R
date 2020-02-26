#' Compile code associated to a user-defined operator with cmake
#' @keywords internal
#' @description
#' The function `compile_code` is a wrapper to call cmake used in the function 
#' [rkeops::compile_formula()].
#' @details
#' The function `compile_code` should only be called in the directory where the 
#' build (i.e. generation of related cmake and so files) will be done.
#' 
#' The corresponging `CMakeLists.txt` file is located in directory `cmake_dir` 
#' given as input argument.
#' @param formula text string, formula defining the new operator.
#' @param var_aliases text string, formated formula input arguments returned by 
#' [rkeops::format_var_aliases()] (specifically `$var_aliases`).
#' @param dllname text string, the name associated to the related shared object 
#' file.
#' @param cmake_dir text string, directory where to find the `CMakeLists.txt` 
#' file.
#' @return an exit code (integer value), 0 if no problem, not 0 otherwise.
#' @seealso [rkeops::compile_formula()], [rkeops::format_var_aliases()]
#' @author Ghislain Durif
#' @export
compile_code <- function(formula, var_aliases, dllname, cmake_dir) {
    
    if(!check_os() | !check_cmake(get_cmake())) return(NULL)
    
    message(paste0('Compiling ', dllname, ' in ', cmake_dir, ':\n',
                   '       formula: ', formula, '\n', 
                   '       aliases: ', var_aliases, '\n', 
                   '       dtype  : ', get_rkeops_option("precision"), '\n...'))
    ## cmake
    cmake_cmd <- paste0(shQuote(get_cmake()), " ", shQuote(cmake_dir), 
                        " -DUSE_CUDA=", 
                        get_rkeops_option("use_cuda_if_possible"),
                        " -DCMAKE_BUILD_TYPE=", 
                        ifelse(!get_rkeops_option("debug"), "Release", "Debug"),
                        " -D__TYPE__=", get_rkeops_option("precision"), 
                        " -DC_CONTIGUOUS=", 
                        as.integer(1-get_rkeops_option("col_major")), 
                        " -DFORMULA_OBJ=", shQuote(formula),
                        " -DVAR_ALIASES=", shQuote(var_aliases),
                        " -Dshared_obj_name=", shQuote(dllname),
                        " -DR_INCLUDE=", shQuote(R.home("include")),
                        " -DRCPP_INCLUDE=", 
                        shQuote(system.file("include", package = "Rcpp")),
                        " -DRCPPEIGEN_INCLUDE=", 
                        shQuote(system.file("include", package = "RcppEigen")),
                        " -DR_LIB=", shQuote(R.home("lib")))
    # FIXME
    # cmake_cmd <- paste0(cmake_cmd,
    #                     " -DcommandLine='", shQuote(cmake_cmd), "'")
    tmp <- system(cmake_cmd, ignore.stdout = !get_rkeops_option("verbosity"), 
                  ignore.stderr = !get_rkeops_option("verbosity"))
    
    ## make
    if(tmp == 0) {
        make_cmd <- paste0("cmake", " --build .", 
                           " --target rkeops", shQuote(dllname), 
                            ifelse(get_rkeops_option("verbosity"), "--VERBOSE=1", ""))
        tmp <- system(make_cmd, ignore.stdout = !get_rkeops_option("verbosity"), 
                      ignore.stderr = !get_rkeops_option("verbosity"))
    }
    
    return(tmp)
}