# function to compile a test function in the binder
# should be executed in "binder/src" directory
compile_test_binder <- function() {
    # Create R/C++ interface
    tmp <- compileAttributes(pkgdir = file.path(get_src_dir(), "binder"))
    if(length(tmp) == 0)
        stop("Issue when generating R/C++ interface")
    # lib name
    dllname <- "test_binder"
    # compile test_binder.cpp
    cmd <- paste0("PKG_CPPFLAGS=", 
                  "\"$(Rscript -e 'Rcpp:::CxxFlags()') ", 
                  "$(Rscript -e 'RcppEigen:::CxxFlags()')\" \\\n",
                  "PKG_LIBS=", 
                  "\"$(Rscript -e 'Rcpp:::LdFlags()')\" \\\n", 
                  R.home("bin"), .Platform$file.sep, "R ",
                  "CMD SHLIB ",
                  "-o ", paste0(dllname, .Platform$dynlib.ext), 
                  " test_binder.cpp RcppExports.cpp")
    tmp <- system(cmd)
    if(tmp != 0)
        stop("Error with compilation")
    # output
    return(dllname)
}