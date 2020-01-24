function [src_dir,bin_folder,precision,verbosity,use_cuda_if_possible] = default_options()
% This function contains some default values loaded 
% when the keops routines are compiled on-the-fly.

src_dir = fileparts([mfilename('fullpath')]);
bin_folder = fullfile(src_dir, 'build', filesep);
precision = 'float';
verbosity = 0;
use_cuda_if_possible = 1; % 0 to force computation on CPU

end
