function [src_dir,build_dir,precision] = default_options()


build_dir = [fileparts([mfilename('fullpath')]),'/build/'];
src_dir = fileparts([mfilename('fullpath')]);
precision = 'float';

end