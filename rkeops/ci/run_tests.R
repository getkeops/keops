# package config
proj_dir <- rprojroot::find_root(".git/index")
pkg_dir <- file.path(proj_dir, "rkeops")

# temp .libPaths
withr::with_temp_libpaths({
    
    # load package
    devtools::load_all(pkg_dir)
    
    # reticulate config
    envname <- "rkeops-ci"
    if(reticulate::virtualenv_exists(envname))
        reticulate::virtualenv_remove(envname, confirm = FALSE)
    reticulate::virtualenv_create(envname)
    reticulate::use_virtualenv(virtualenv = envname, required = TRUE)
    reticulate::py_config()
    
    # install requirements
    install_rkeops()
    
    # check
    check_rkeops()
    
    # run tests
    devtools::test(pkg_dir, reporter = c('Progress', 'fail'))
})
