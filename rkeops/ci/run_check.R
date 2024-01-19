# package config
proj_dir <- rprojroot::find_root(".git/index")
pkg_dir <- file.path(proj_dir, "rkeops")

# temp .libPaths
withr::with_temp_libpaths({
    # reticulate config
    envname <- "rkeops-ci"
    if(reticulate::virtualenv_exists(envname))
        reticulate::virtualenv_remove(envname, confirm = FALSE)
    reticulate::virtualenv_create(envname)
    reticulate::virtualenv_install(envname, "pykeops")
    reticulate::use_virtualenv(virtualenv = envname, required = TRUE)
    reticulate::py_config()
    
    # load package
    devtools::install(pkg_dir, upgrade = TRUE)
    
    # load package
    library(rkeops)
    
    # run check
    devtools::check(pkg_dir, error_on = "error")
})