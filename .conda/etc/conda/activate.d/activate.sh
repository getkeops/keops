export TORCH_USE_RTLD_GLOBAL=YES # see https://github.com/getkeops/keops/issues/59
unset CXXFLAGS # flags set by the conda compiler packages are problematic for keops
