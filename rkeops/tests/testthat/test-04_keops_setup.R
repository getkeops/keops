test_that("def_pykeops_build_dir", {
    skip_if_no_python()
    skip_if_no_pykeops()
    
    withr::with_options(list(rkeops = NULL), {
        set_rkeops_options(list(cache_dir = testing_cache_dir))
    
        res <- def_pykeops_build_dir()
        checkmate::expect_directory(res)
    })
})

test_that("set_pykeops_verbosity", {
    skip_if_no_python()
    skip_if_no_pykeops()
    
    withr::with_options(list(rkeops = NULL), {
        set_rkeops_options(list(cache_dir = testing_cache_dir))
        
        # enable verbosity
        rkeops_enable_verbosity()
        set_pykeops_verbosity()
        
        expect_equal(pykeops$verbose, TRUE)
        
        # disable verbosity
        rkeops_disable_verbosity()
        set_pykeops_verbosity()
        
        expect_equal(pykeops$verbose, FALSE)
    })
})

test_that("set_pykeops_build_dir", {
    skip_if_no_python()
    skip_if_no_pykeops()
    
    withr::with_options(list(rkeops = NULL), {
        set_rkeops_options(list(cache_dir = testing_cache_dir))
        
        set_pykeops_build_dir()
        expect_equal(def_pykeops_build_dir(), pykeops$get_build_folder())
        
    })
})


test_that("setup_pykeops", {
    skip_if_no_python()
    skip_if_no_pykeops()
    
    withr::with_options(list(rkeops = NULL), {
        set_rkeops_options(list(cache_dir = testing_cache_dir))
        setup_pykeops()
        expect_equal(def_pykeops_build_dir(), pykeops$get_build_folder())
        expect_equal(
            as.logical(get_rkeops_options("verbosity")), pykeops$verbose)
    })
})
