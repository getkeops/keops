# helper function to skip tests if Python is not available on the system
skip_if_no_python <- function() {
    have_python <- reticulate::py_available()
    if(!have_python) skip("Python not available on system for testing")
}

# helper function to skip tests if `keopscore` is not available
skip_if_no_keopscore <- function() {
    skip_if_no_python()
    have_keopscore <- rkeops:::check_keopscore()
    if(!have_keopscore) skip("'keopscore' not available for testing")
}

# helper function to skip tests if `pykeops` is not available
skip_if_no_pykeops <- function() {
    skip_if_no_python()
    have_pykeops <- rkeops:::check_pykeops()
    if(!have_pykeops) skip("'pykeops' not available for testing")
}

# helper function to skip tests if not interactive mode
skip_if_not_interactive <- function() {
    if(!interactive()) skip("Test only run in interactive mode")
}
