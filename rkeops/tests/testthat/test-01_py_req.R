test_that("install_rkeops", {
    skip_if_no_python()
    
    withr::with_options(list(rkeops = NULL), {
        set_rkeops_options()
        setup_pyreq()
        expect_true(reticulate::py_available("pykeops"))
    })
})
