test_that("get_os", {
    checkmate::expect_choice(get_os(), choices = c("linux", "macos", "windows"))
})

test_that("msg_warn_error", {
    msg <- "test message"
    expect_message(msg_warn_error(msg, type = "msg"), msg)
    expect_message(msg_warn_error(msg, type = "msg", startup = TRUE), msg)
    expect_warning(msg_warn_error(msg, type = "warn"), msg)
    expect_error(msg_warn_error(msg, type = "error"), msg)
    # bad input
    expect_error(msg_warn_error(msg, type = "print"))
})

test_that("random_varname", {
    
    expect_equal(random_varname(prefix = "", len = 0), "")
    
    res <- random_varname(prefix = "", len = 10)
    checkmate::expect_string(res, n.char = 10, pattern = "[A-Za-z0-9]{10}")
    
    prefix <- "prefix"
    res <- random_varname(prefix, len = 10)
    checkmate::expect_string(
        res, n.char = 10 + stringr::str_length(prefix),
        pattern = stringr::str_c("^", prefix, "[A-Za-z0-9]{10}")
    )
})
