#' Generatation of hash values from a text strings
#' @keywords internal
#' @description
#' This function uses a sha256 algorithm to generate 
#' a hash from a text string.
#' @details
#' This function is used to generate unique identifier 
#' from text formulae that is used to store 
#' compiled associated code.
#' @author Ghislain Durif
#' @param str a text string.
#' @return the associated hash value as a character 
#' string.
#' @importFrom openssl sha256
#' @export
string2hash <- function(str) {
    out <- sha256(str)
    return(out)
}
