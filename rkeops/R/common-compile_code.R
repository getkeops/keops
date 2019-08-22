#' Compile code
#' @description
#' FIXME
#' @details
#' FIXME
#' @author Ghislain Durif
#' @export
compile_code <- function(formula, var_aliases, dllname, cmake_dir) {
    # FIXME
    
    ## cmake
    cmake_cmd <- paste0(get_cmake(), " ", cmake_dir, 
                        " -DUSE_CUDA=", get_rkeops_option("use_cuda_if_possible"),
                        " -DCMAKE_BUILD_TYPE=Release",
                        " -D__TYPE__=", get_rkeops_option("precision"), 
                        " -DFORMULA_OBJ=", formula,
                        " -DVAR_ALIASES=", var_aliases,
                        " -Dshared_obj_name=", dllname,
                        " -DR_INCLUDE=", R.home("include"),
                        " -DRCPP_INCLUDE=", 
                        system.file("include", package = "Rcpp"),
                        " -DRCPPEIGEN_INCLUDE=", 
                        system.file("include", package = "RcppEigen"),
                        " -DR_LIB=", R.home("lib"))
    cmake_cmd <- paste0(cmake_cmd, 
                        " -DcommandLine='", cmake_cmd, "'")
    tmp <- system(cmake_cmd)
    
    ## make
    if(tmp == 0) {
        make_cmd <- paste0("cmake", " --build .", 
                           " --target rkeops", dllname, " --" , " VERBOSE=1")
        tmp <- system(make_cmd)
    }
    
    return(tmp)
}