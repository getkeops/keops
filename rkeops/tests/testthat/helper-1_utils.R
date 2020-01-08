# function to compile a test function
# should be executed in the "<rkeops_dir>/src" directory
compile_test_function <- function() {
    # Create R/C++ interface
    tmp <- compileAttributes(pkgdir = get_pkg_dir())
    if(length(tmp) == 0)
        stop("Issue when generating R/C++ interface")
    # lib name
    dllname <- "test_function"
    # compile test_binder.cpp
    cmd <- paste0("PKG_CPPFLAGS=-I", 
                  shQuote(system.file("include", package = "Rcpp")), " ",  
                  shQuote(paste0(R.home("bin"), .Platform$file.sep, "R")),
                  " CMD SHLIB ",
                  "-o ", shQuote(file.path(get_build_dir(), 
                                           paste0(dllname, 
                                                  .Platform$dynlib.ext))), 
                  " rkeops.cpp RcppExports.cpp")
    tmp <- system(cmd, ignore.stdout = TRUE, ignore.stderr = TRUE)
    if(tmp != 0)
        stop("Error with compilation")
    # output
    return(dllname)
}
