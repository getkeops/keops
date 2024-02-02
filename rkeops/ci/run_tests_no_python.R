# package config
proj_dir <- rprojroot::find_root(".git/index")
pkg_dir <- file.path(proj_dir, "rkeops")

# temp .libPaths
withr::with_temp_libpaths({
    
    # load package
    devtools::load_all(pkg_dir)
    
    # run tests
    devtools::test(pkg_dir, reporter = c('Progress', 'fail'))
})
