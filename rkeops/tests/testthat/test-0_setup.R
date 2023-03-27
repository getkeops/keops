test_that("skip_if_no_python", {
    # Test that a skip happens
    if(reticulate::py_available()) {
        # no skip
        expect_condition(skip_if_no_python(), NA, class = "skip")
    } else {
        # skip
        expect_condition(skip_if_no_python(), class = "skip")
    }
})

test_that("skip_if_no_keopscore", {
    skip("WRITE ME")
    skip_if_no_python()
})

test_that("skip_if_no_pykeops", {
    skip("WRITE ME")
    skip_if_no_python()
})

test_that("skip_if_not_interactive", {
    # Test that a skip happens
    if(!interactive()) {
        # no skip
        expect_condition(skip_if_not_interactive(), NA, class = "skip")
    } else {
        # skip
        expect_condition(skip_if_not_interactive(), class = "skip")
    }
})
