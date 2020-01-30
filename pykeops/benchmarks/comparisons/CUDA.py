"""
Handcrafted CUDA
=========================
"""

###################################
# Blabla
#

!nvcc -O3 -D_FORCE_INLINES --use_fast_math --compiler-options=-fPIC conv_specific.cu -o conv_specific
!./conv_specific
