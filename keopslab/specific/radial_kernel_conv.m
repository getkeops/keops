function [res] = radial_kernel_conv(x,y,b,s,kernel_type)
% This function computes the convolution with a radial kernel:
%
%   \sum_j K_s(x_i,y_j) * b_j
%
% Inputs
%   x : a (nx x d) matrix
%   y : a (ny x d) matrix
%   b : a (ny x e) matrix
%   s : a real number (kernel width)
%   kernel_type : a string. ('gaussian', 'cauchy', 'laplacian', 'inverse_multiquadric')
%
% Output
%   res : a real number
%

if ~(exist(['conv.',mexext],'file')==3)
    compile_routine_conv([]);
end

res = conv(x,y,b,s,kernel_type);

end
