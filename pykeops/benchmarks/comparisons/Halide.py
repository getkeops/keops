"""
Halide
====================

"""

#########################
# Blabla
#


g++ gauss_conv_halide.cpp -g -std=c++11 -I halide/tutorial -I halide/include -I halide/tools -L halide/bin -lHalide -lpthread -ldl -o gauss_conv_halide
LD_LIBRARY_PATH=halide/bin ./gauss_conv_halide 10000


