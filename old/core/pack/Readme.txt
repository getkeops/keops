/*
 * To differentiate automatically our code at compilation time, KeOps relies heavily on
 * variadic std=c++11 syntax. The idea is to use the compiler as a "formula/graph-processing engine",
 * and we need it to process tree structures and lists of variables.
 * This is achieved using the recursive variadic templating.
 * We define the following "container/symbolic" templates:
 * - univpack,    which acts as a list of symbolic types
 * - pack,        which acts more specifically as a list of vectors of known sizes
 * - CondType,    which acts as a symbolic conditional statement
 * - ConcatPacks, which is a concatenation operator for packs
 *
 */