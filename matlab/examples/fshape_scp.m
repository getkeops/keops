nx = 4; d= 3; ny = 5;

if 0
    center_faceX = randn(nx,d);
    signalX = randn(nx,1);
    normalsX = randn(nx,d);
    center_faceY = randn(ny,d);
    signalY = randn(ny,1);
    normalsY = randn(ny,d);
else
    center_faceX     = linspace(.5,2,nx)' * linspace(0,1,d)
    normalsX         = linspace(-1,2,nx)' * linspace(1,2,d)
    signalX         = -linspace(1,2,nx)'
    center_faceY     = -linspace(1,3,ny)' * linspace(1,2,d)
    normalsY         = -linspace(1,2,ny)' * ones(1,d)
    signalY         = linspace(1,2,ny)'
end
kernel_size_geom = 1;
kernel_size_signal = 1;
kernel_size_sphere =pi;


kernel_gaussian = @(r2,s) exp(-r2 / s^2);
dkernel_gaussian = @(r2,s) -exp(-r2/s^2)/l^2;

kernel_cauchy = @(r2,s)  1 ./ (1 + (r2/s^2));
dkernel_cauchy = @(r2,s) -1 ./ (s^2 * (1 + (r2/s^2)) .^2);

kernel_linear = @(prs,~) prs;
dkernel_linear = @(prs,~) ones(size(prs)); 
        
kernel_gaussian_oriented = @(prs,s)  exp( (-2 + 2*prs) / s^2);
dkernel_gaussian_oriented = @(prs,s) 2 * exp( (-2 +2*prs) / s^2) / s^2;
        
kernel_binet = @(prs,~) prs .^2;
dkernel_binet = @(prs,~)  2*prs;
        
kernel_gaussian_unoriented = @(prs,s)  exp( (-2 + 2 * prs.^2) / s^2);
dkernel_gaussian_unoriented = @(prs,s) 4 * x .* exp( (-2 + 2 *prs.^2) /s^2) / s^2;


kernel_geom = 'cauchy';
kernel_signal = 'gaussian';
kernel_sphere = 'binet';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                  CUDA                                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath('../matkp/mex')
eval(['cuda_fshape_scp=@cuda_fshape_scp_',lower(kernel_geom),lower(kernel_signal),lower(kernel_sphere),';']);
        
%prs(x,y) =
XY= cuda_fshape_scp(center_faceX',center_faceY',signalX',signalY',normalsX',normalsY',kernel_size_geom,kernel_size_signal,kernel_size_sphere);
res_cuda =  sum(XY);

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

% Kernels
eval(['radial_function_geom=kernel_',lower(kernel_geom)  ,';']);
eval(['radial_function_signal=kernel_',lower(kernel_signal),';']);
eval(['radial_function_sphere=kernel_',lower(kernel_sphere) ,';']);

Kernel_geomXY = radial_function_geom(distance_center_faceXY,kernel_size_geom);
Kernel_signalXY = radial_function_signal(distance_signalXY,kernel_size_signal);
Kernel_sphereXY = radial_function_sphere(oriented_angle_normalsXY,kernel_size_sphere);

%prs(x,y) =
sum((norm_normalsX * norm_normalsY') .* Kernel_geomXY .* Kernel_signalXY .* Kernel_sphereXY,2)'
res_matlab = sum(sum((norm_normalsX * norm_normalsY') .* Kernel_geomXY .* Kernel_signalXY .* Kernel_sphereXY));

fprintf('relative errror between cuda and matlab version: %g\n',abs( (res_matlab - res_cuda) ./ res_matlab ))
