% This example script provide a comparison between various version of Cuda
% code used to compute distance between shapes. The script works only with 
% a GPU.


path_to_lib = [fileparts(mfilename('fullpath')), filesep, '..'];
addpath(genpath(path_to_lib))

nx = 15; d= 3; ny = 51;

center_faceX = randn(nx,d);
signalX = randn(nx,1);
normalsX = 1+0*randn(nx,d);
center_faceY = randn(ny,d);
signalY = randn(ny,1);
normalsY = randn(ny,d);

kernel_size_geom = 1.23124;
kernel_size_signal = 1.232;
kernel_size_sphere =pi;


kernel_gaussian = @(r2,s) exp(-r2 / s^2);
dkernel_gaussian = @(r2,s) -exp(-r2/s^2)/s^2;

kernel_cauchy = @(r2,s)  1 ./ (1 + (r2/s^2));
dkernel_cauchy = @(r2,s) -1 ./ (s^2 * (1 + (r2/s^2)) .^2);

kernel_linear = @(prs,~) prs;
dkernel_linear = @(prs,~) ones(size(prs)); 
        
kernel_gaussian_oriented = @(prs,s)  exp( (-2 + 2*prs) / s^2);
dkernel_gaussian_oriented = @(prs,s) 2 * exp( (-2 +2*prs) / s^2) / s^2;
        
kernel_binet = @(prs,~) prs .^2;
dkernel_binet = @(prs,~)  2*prs;
        
kernel_gaussian_unoriented = @(prs,s)  exp( (-2 + 2 * prs.^2) / s^2);
dkernel_gaussian_unoriented = @(prs,s) 4 * prs .* exp( (-2 + 2 *prs.^2) /s^2) / s^2;


opt.kernel_geom = 'cauchy';
opt.kernel_signal = 'gaussian';
opt.kernel_sphere = 'gaussian_oriented';


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                  CUDA                                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
%prs(x,y) =
res_cuda = shape_scp(center_faceX,center_faceY,signalX,signalY,normalsX,normalsY,kernel_size_geom,kernel_size_signal,kernel_size_sphere,opt);
res_cuda_dx = shape_scp_dx(center_faceX,center_faceY,signalX,signalY,normalsX,normalsY,kernel_size_geom,kernel_size_signal,kernel_size_sphere,opt);
res_cuda_dxi = shape_scp_dxi(center_faceX,center_faceY,signalX,signalY,normalsX,normalsY,kernel_size_geom,kernel_size_signal,kernel_size_sphere,opt);
res_cuda_df = shape_scp_df(center_faceX,center_faceY,signalX,signalY,normalsX,normalsY,kernel_size_geom,kernel_size_signal,kernel_size_sphere,opt);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                MATLAB                                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Compute norms of the normals
norm_normalsX = sqrt(sum(normalsX .^2,2));
norm_normalsY = sqrt(sum(normalsY .^2,2));

% Compute unit normals
unit_normalsX = normalsX ./  repmat(norm_normalsX,1,size(normalsX,2));
unit_normalsY = normalsY ./  repmat(norm_normalsY,1,size(normalsY,2));

%compute squared distances and angles
distance_signalXY = (repmat(signalX,1,ny)-repmat(signalY',nx,1)).^2;
distance_center_faceXY=zeros(nx,ny);
oriented_angle_normalsXY = zeros(nx,ny);
        
for l=1:d
    distance_center_faceXY = distance_center_faceXY+(repmat(center_faceX(:,l),1,ny)-repmat(center_faceY(:,l)',nx,1)).^2;
    oriented_angle_normalsXY = oriented_angle_normalsXY + (repmat(unit_normalsX(:,l),1,ny).*repmat(unit_normalsY(:,l)',nx,1));
end

% kernels
eval(['radial_function_geom=kernel_',lower(opt.kernel_geom)  ,';']);
eval(['dradial_function_geom=dkernel_',lower(opt.kernel_geom)  ,';']);
eval(['radial_function_signal=kernel_',lower(opt.kernel_signal),';']);
eval(['dradial_function_signal=dkernel_',lower(opt.kernel_signal),';']);
eval(['radial_function_sphere=kernel_',lower(opt.kernel_sphere) ,';']);
eval(['dradial_function_sphere=dkernel_',lower(opt.kernel_sphere) ,';']);

kernel_geomXY = radial_function_geom(distance_center_faceXY,kernel_size_geom);
dkernel_geomXY = dradial_function_geom(distance_center_faceXY,kernel_size_geom);
kernel_signalXY = radial_function_signal(distance_signalXY,kernel_size_signal);
dkernel_signalXY = dradial_function_signal(distance_signalXY,kernel_size_signal);
kernel_sphereXY = radial_function_sphere(oriented_angle_normalsXY,kernel_size_sphere);
dkernel_sphereXY = dradial_function_sphere(oriented_angle_normalsXY,kernel_size_sphere);

%prs(x,y) =
%sum((norm_normalsX * norm_normalsY') .* kernel_geomXY .* kernel_signalXY .* kernel_sphereXY,2)'
res_matlab = sum(sum((norm_normalsX * norm_normalsY') .* kernel_geomXY .* kernel_signalXY .* kernel_sphereXY));
for l=1:d 
    res_matlab_dx(:,l) = 2* sum( (repmat(center_faceX(:,l),1,ny)-repmat(center_faceY(:,l)',nx,1)) .* (norm_normalsX * norm_normalsY') .* dkernel_geomXY .* kernel_signalXY .* kernel_sphereXY,2);
end

MXY = kernel_geomXY .* kernel_signalXY .* dkernel_sphereXY;
mXY = kernel_geomXY .* kernel_signalXY .* kernel_sphereXY;
res_matlab_dxi =  repmat(mXY * norm_normalsY  - (MXY .* oriented_angle_normalsXY  ) * norm_normalsY,1,d) .* unit_normalsX + MXY * normalsY;

res_matlab_df = 2 * sum( (repmat(signalX,1,ny) - repmat(signalY',nx,1)) .* (norm_normalsX * norm_normalsY') .* kernel_geomXY .* dkernel_signalXY .* kernel_sphereXY, 2);

fprintf('Relative error between cuda and matlab version: %g\n',abs( (res_matlab - res_cuda) ./ res_matlab ))
fprintf('Relative error between cuda and matlab version: %g\n',sum(sum(abs( (res_matlab_dx - res_cuda_dx) ./ res_matlab_dx ))))
fprintf('Relative error between cuda and matlab version: %g\n',sum(sum(abs( (res_matlab_dxi - res_cuda_dxi) ./ res_matlab_dxi ))))
fprintf('Relative error between cuda and matlab version: %g\n',sum(sum(abs( (res_matlab_df - res_cuda_df) ./ res_matlab_df ))))
