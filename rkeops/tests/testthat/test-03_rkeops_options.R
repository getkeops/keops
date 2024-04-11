test_that("def_rkeops_options", {
    res <- def_rkeops_options()
    
    expect_equal(class(res), "rkeops_options")
    checkmate::expect_list(res, len = 6)
    checkmate::expect_set_equal(
        names(res), 
        c("backend", "device_id", "precision", "verbosity", "debug", 
          "cache_dir"))
    checkmate::expect_choice(res$backend, c("CPU", "GPU"))
    checkmate::expect_integerish(res$device_id)
    checkmate::expect_choice(res$precision, c("float32", "float64"))
    checkmate::expect_choice(res$verbosity, c(0, 1))
    checkmate::expect_choice(res$debug, c(0, 1))
    checkmate::expect_directory(res$cache_dir)
    
    res <- def_rkeops_options(
        backend = "CPU", device_id = -1, precision = "float32",
        verbosity = FALSE, debug = FALSE, cache_dir = getwd())
    
    expect_equal(class(res), "rkeops_options")
    checkmate::expect_list(res, len = 6)
    checkmate::expect_set_equal(
        names(res), 
        c("backend", "device_id", "precision", "verbosity", "debug", 
          "cache_dir"))
    checkmate::expect_choice(res$backend, c("CPU", "GPU"))
    checkmate::expect_integerish(res$device_id)
    checkmate::expect_choice(res$precision, c("float32", "float64"))
    checkmate::expect_choice(res$verbosity, c(0, 1))
    checkmate::expect_choice(res$debug, c(0, 1))
    expect_equal(res$cache_dir, getwd())
    
    expect_error(def_rkeops_options(backend = "TPU"))
    expect_error(def_rkeops_options(device_id = 2.5))
    expect_error(def_rkeops_options(precision = "integer"))
    expect_error(def_rkeops_options(verbosity = "FALSE"))
    expect_error(def_rkeops_options(debug = "FALSE"))
    expect_error(def_rkeops_options(cache_dir = "/not/existing/dir"))
})

test_that("get_rkeops_options", {
    
    # no rkeops options
    withr::with_options(list(rkeops = NULL), {
        expect_error(get_rkeops_options())
    })
    
    # work with clean options
    withr::with_options(list(rkeops = NULL), {
        # check empty options
        expect_equal(getOption("rkeops"), NULL)
        
        # set default options
        set_rkeops_options()
        
        # get all options
        res <- get_rkeops_options()
        expect_equal(res, def_rkeops_options())
        
        # get specific options
        res <- get_rkeops_options("backend")
        checkmate::expect_character(res, len = 1)
        checkmate::expect_choice(res, c("CPU", "GPU"))
        
        res <- get_rkeops_options(c("backend", "precision"))
        checkmate::expect_list(res, len = 2)
        expect_equal(names(res), c("backend", "precision"))
        checkmate::expect_choice(res$backend, c("CPU", "GPU"))
        checkmate::expect_choice(res$precision, c("float32", "float64"))
    })
})

test_that("set_rkeops_options", {
    
    withr::with_options(list(rkeops = NULL), {
        # default options
        expect_error(set_rkeops_options(), NA)
        
        # check setup
        expect_equal(getOption("rkeops"), def_rkeops_options())
        
        # current cache dir
        checkmate::expect_directory(getOption("rkeops")$cache_dir)
    })
    
    withr::with_options(list(rkeops = NULL), {
        # set specific options
        expect_error(set_rkeops_options(list(backend = "GPU")), NA)
        res <- getOption("rkeops")
        # verify specific options
        expect_equal(res$backend, "GPU")
        # verify other options
        default_options <- def_rkeops_options()
        default_options["backend"] <- "GPU"
        expect_equal(res, default_options)
    })
        
    withr::with_options(list(rkeops = NULL), {
        # wrong input
        expect_error(set_rkeops_options("backend"))
    })
})

test_that("rkeops_use_gpu", {
    
    withr::with_options(list(rkeops = NULL), {
        # use GPU
        rkeops_use_gpu()
        # check
        res <- get_rkeops_options()
        expect_equal(res$backend, "GPU")
        expect_equal(res$device_id, -1)
    })
    
    withr::with_options(list(rkeops = NULL), {
        expect_error(rkeops_use_gpu(device = "0"))
    })
})

test_that("rkeops_use_cpu", {
    
    withr::with_options(list(rkeops = NULL), {
        # use CPU
        rkeops_use_cpu()
        # check
        res <- get_rkeops_options()
        expect_equal(res$backend, "CPU")
    })
    
    withr::with_options(list(rkeops = NULL), {
        expect_error(rkeops_use_cpu(ncore = "0"))
    })
})

test_that("rkeops_use_float32", {
    
    withr::with_options(list(rkeops = NULL), {
        # use float32
        rkeops_use_float32()
        # check
        res <- get_rkeops_options()
        expect_equal(res$precision, "float32")
    })
})

test_that("rkeops_use_float64", {
    
    withr::with_options(list(rkeops = NULL), {
        # use float64
        rkeops_use_float64()
        # check
        res <- get_rkeops_options()
        expect_equal(res$precision, "float64")
    })
})

test_that("rkeops_enable_verbosity", {
    
    withr::with_options(list(rkeops = NULL), {
        # enable verbosity
        rkeops_enable_verbosity()
        # check
        res <- get_rkeops_options()
        expect_equal(res$verbosity, 1)
    })
})

test_that("rkeops_disable_verbosity", {
    
    withr::with_options(list(rkeops = NULL), {
        # disable verbosity
        rkeops_disable_verbosity()
        # check
        res <- get_rkeops_options()
        expect_equal(res$verbosity, 0)
    })
})

test_that("get_rkeops_cache_dir", {
    res <- get_rkeops_cache_dir()
    checkmate::expect_string(res)
    checkmate::expect_directory(res)
})

test_that("set_rkeops_cache_dir", {
    
    withr::with_options(list(rkeops = NULL), {
        # set options
        set_rkeops_options()
        # default dir
        set_rkeops_cache_dir(verbose = FALSE)
        # check cache dir
        checkmate::expect_directory(getOption("rkeops")$cache_dir)
    })
    
    withr::with_options(list(rkeops = NULL), {
        # set options
        set_rkeops_options()
        # specific dir
        set_rkeops_cache_dir(getwd(), verbose = FALSE)
        
        # check cache dir
        checkmate::expect_directory(getOption("rkeops")$cache_dir)
        expect_equal(getOption("rkeops")$cache_dir, getwd())
    })
})

test_that("default_rkeops_cache_dir", {
    
    res <- default_rkeops_cache_dir()
    checkmate::expect_directory(res)
    
    if(getRversion() >= "4.0") {
        expect_equal(
            res, tools::R_user_dir(package = "rkeops", which = "cache"))
    } else {
        expect_equal(
            res, file.path(tempdir(), "rkeops")
        )
    }
})
