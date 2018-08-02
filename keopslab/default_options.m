function [src_dir,build_dir,precision,verbosity,use_cuda_if_possible] = default_options()
% This function contains some default values loaded 
% when the keops routines are compiled on-the-fly.

build_dir = [fileparts([mfilename('fullpath')]),'/build/'];
src_dir = fileparts([mfilename('fullpath')]);
precision = 'float';
verbosity = 0;
use_cuda_if_possible = 1; % 0 to force computation on CPU

end
