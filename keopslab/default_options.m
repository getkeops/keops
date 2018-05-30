function [src_dir,build_dir,precision,verbosity] = default_options()
% This function contains some default values loaded 
% when the keops routines are compile on-the-fly.

build_dir = [fileparts([mfilename('fullpath')]),'/build/'];
src_dir = fileparts([mfilename('fullpath')]);
precision = 'float';
verbosity = 1;

end
