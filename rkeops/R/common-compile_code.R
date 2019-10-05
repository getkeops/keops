#' Compile code
#' @description
#' FIXME
#' @details
#' FIXME
#' @author Ghislain Durif
#' @export
compile_code <- function(formula, var_aliases, dllname, cmake_dir) {
    
    ## cmake
    cmake_cmd <- paste0(shQuote(get_cmake()), " ", shQuote(cmake_dir), 
                        " -DUSE_CUDA=", 
                        get_rkeops_option("use_cuda_if_possible"),
                        " -DCMAKE_BUILD_TYPE=Release",
                        " -D__TYPE__=", get_rkeops_option("precision"), 
                        " -DC_CONTIGUOUS=0", 
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
    # " -DCMAKE_BUILD_TYPE=Debug")
    # FIXME
    # cmake_cmd <- paste0(cmake_cmd,
    #                     " -DcommandLine='", shQuote(cmake_cmd), "'")
    tmp <- system(cmake_cmd)
    
    ## make
    if(tmp == 0) {
        make_cmd <- paste0("cmake", " --build .", 
                           " --target rkeops", shQuote(dllname), 
                           " --" , " VERBOSE=1")
        tmp <- system(make_cmd)
    }
    
    return(tmp)
}