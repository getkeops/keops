# package config
proj_dir <- rprojroot::find_root(".git/index")
pkg_dir <- file.path(proj_dir, "rkeops")

# temp .libPaths
withr::with_temp_libpaths({
    
    # load package
    devtools::load_all(pkg_dir)
    
    # reticulate config
    reticulate::py_config()
    
    # run check
    devtools::check(pkg_dir, error_on = "error")
})