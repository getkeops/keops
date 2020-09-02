# test compilation of rkeops operators
project_root_dir <- system("git rev-parse --show-toplevel", intern=TRUE)
devtools::load_all(file.path(project_root_dir, "rkeops"))

set_rkeops_options()
# matrix product then sum
formula = "Sum_Reduction((x|y), 0)"
args = c("x=Vi(3)", "y=Vj(3)")

var_aliases <- format_var_aliases(args)$var_aliases
dllname <- "test_compile_code_dll"

## cmake src dir
cmake_dir <- dirname(get_rkeops_option("src_dir"))

## get user current working directory
current_directory <- getwd()

## generate binder
tmp <- compileAttributes(pkgdir = file.path(get_src_dir(),
                                            "binder"))

## move to build directory
tmp_build_dir <- file.path(get_rkeops_option("build_dir"), dllname)
if(!dir.exists(tmp_build_dir)) dir.create(tmp_build_dir)
setwd(tmp_build_dir)

## compiling (call to cmake)
return_status <- tryCatch(compile_code(formula, var_aliases,
                                       dllname, cmake_dir),
                          error = function(e) {print(e); return(NULL)})

## move back to user working directory
setwd(current_directory)

## test compilation return status
expect_equal(return_status, 0)

## check compilation
test_binder <- tryCatch(load_dll(path = tmp_build_dir,
                                 dllname = paste0("librkeops", dllname),
                                 object = "test_binder"),
                        error = function(e) {print(e); return(NULL)})
expect_false(is.null(test_binder))
expect_true(test_binder() == 1)

message("DONE")