test_that("get_os", {
    expect_choice(get_os(), choices = c("linux", "macos", "windows"))
})

test_that("msg_warn_error", {
    msg <- "test message"
    expect_message(msg_warn_error(msg, type = "msg"), msg)
    expect_warning(msg_warn_error(msg, type = "warn"), msg)
    expect_error(msg_warn_error(msg, type = "error"), msg)
    # bad input
    expect_error(msg_warn_error(msg, type = "print"))
})
