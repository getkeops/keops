context("Starting")

test_that("helloWorld", {
    expect_output(helloWorld(), "Hello user of KeOps")
})
