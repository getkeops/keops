test_that("install_rkeops", {
    skip_if_no_python()
    set_rkeops_options()
    
    expect_error(install_rkeops(), NA)
})
