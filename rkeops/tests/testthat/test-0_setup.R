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
    skip_if_no_python()
    if(check_keopscore(warn=FALSE)) {
        # no skip
        expect_condition(skip_if_no_keopscore(), NA, class = "skip")
    } else {
        # skip
        expect_condition(skip_if_no_keopscore(), class = "skip")
    }
})

test_that("skip_if_no_pykeops", {
    skip_if_no_python()
    if(check_pykeops(warn=FALSE)) {
        # no skip
        expect_condition(skip_if_no_pykeops(), NA, class = "skip")
    } else {
        # skip
        expect_condition(skip_if_no_pykeops(), class = "skip")
    }
})

test_that("skip_if_not_interactive", {
    # Test that a skip happens
    if(interactive()) {
        # no skip
        expect_condition(skip_if_not_interactive(), NA, class = "skip")
    } else {
        # skip
        expect_condition(skip_if_not_interactive(), class = "skip")
    }
})
