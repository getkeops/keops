test_that("install_rkeops", {
    skip_if_no_python()
    
    withr::with_options(list(rkeops = NULL), {
        set_rkeops_options(list(cache_dir = testing_cache_dir))
        expect_error(install_rkeops(), NA)
    })
})
