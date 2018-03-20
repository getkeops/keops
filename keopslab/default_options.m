function [src_dir,build_dir,precision] = default_options()
% This function contains some default value loaded 
% when the keops routines are compile on-the-fly.

build_dir = [fileparts([mfilename('fullpath')]),'/build/'];
src_dir = fileparts([mfilename('fullpath')]);
precision = 'float';

end
