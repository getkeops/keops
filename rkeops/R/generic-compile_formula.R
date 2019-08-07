#' Compile code
#' @description
#' FIXME
#' @details
#' FIXME
#' @author Ghislain Durif
#' @export
compile_formula <- function(formula, var_aliases, hash) {
    
    # FIXME
    
    cmd_cmake <- paste0("cmake",
                        " -DFORMULA_OBJ=", formula,
                        " -DVAR_ALIASES=", var_aliases,
                        " -Dshared_obj_name=", dllname)
    
    
    # cmdline = [cmake, ' ', src_dir , ' -DUSE_CUDA=', num2str(use_cuda_if_possible), ...
    #            ' -DCMAKE_BUILD_TYPE=Release', ' -D__TYPE__=', precision, ...
    #            ' -DMatlab_ROOT_DIR="', matlabroot, '" ', cmd_cmake];
}