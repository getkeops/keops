function tests = generic_test
    tests = functiontests(localfunctions);
end


function setupOnce(testCase)  % do not change function name
    % set a new path, for example
    path_to_lib = [fileparts(mfilename('fullpath')) , filesep, '..']
    addpath(genpath(path_to_lib))
    
end


function res= allclose(a,b)

    atol = 1e-6;
    rtol = 1e-4;
    res = all( abs(a(:)-b(:)) <= atol+rtol*abs(b(:)) );
end

function res = squmatrix_distance(x,y) 
    res = sum( (repmat(reshape(x,size(x,1),1,size(x,2)),1,size(y,1),1)  - repmat(reshape(y,1,size(y,1),size(y,2)),size(x,1),1,1)) .^2,3);
end


%-----------------------------------------------%
%   Test 1: standard expression
%-----------------------------------------------%
function test_standard_expression(testCase)

    Nx = 50; Ny = 20;
    x = randn(3,Nx); y = randn(3,Ny); b = randn(3,Ny); u = randn(4,Nx); v = randn(4,Ny);
    p = .25;

    % Kernel with KeOps
    F = keops_kernel('x=Vi(3)','y=Vj(3)','u=Vi(4)','v=Vj(4)','b=Vj(3)', 'p=Pm(1)', 'Sum_Reduction(Square((u|v))*Exp(-p*SqNorm2(x-y))*b,0)');
    g = F(x,y,u,v,b,p);
    
    % Compare with matlab
    uv = u' * v;
    g0 = ((uv .^2 .* exp(-p*squmatrix_distance(x',y'))) *b')';
    
    assert( allclose(g,g0) == 1 )
end


%-----------------------------------------------%
%   Test 2: standard expression with gradient
%-----------------------------------------------%
function test_standard_expression_gradient(testCase)
    Nx = 50; Ny = 20;
    x = randn(3,Nx); y = randn(3,Ny); a = randn(3,Nx); b = randn(3,Ny);
    p = .25;
    % Kernel with KeOps
    F1 = keops_kernel('x=Vi(0,3)','y=Vj(1,3)','b=Vj(2,3)','a=Vi(3,3)', 'p=Pm(4,1)', 'Sum_Reduction(Grad(Exp(-p*SqNorm2(x-y))*b,x,a),0)');
    g1 = F1(x,y,b,a,p);
    
    % Compare with matlab
    g2 = zeros(3,Nx);
    dkernel_geomXY = - exp(-p*squmatrix_distance(x',y')) * p;
    for l=1:3 
        g2(l,:) = 2 * sum( a .* ( b * ( (repmat(x(l,:)',1,Ny)-repmat(y(l,:),Nx,1))  .* dkernel_geomXY)' ),1) ;
    end
    
    assert( allclose(g1,g2)==1 )
end

%-----------------------------------------------%
%   Test 3: gradient
%-----------------------------------------------%
function test_gradient(testCase)
    Nx = 50; Ny = 20;
    a = randn(3,Nx); x = randn(3,Nx); y = randn(3,Ny); b = randn(3,Ny);
    p = .25;

    F0 = keops_kernel('x=Vi(3)','y=Vj(3)','b=Vj(3)', 'p=Pm(1)', 'Sum_Reduction(Exp(-p*SqNorm2(x-y))*b,0)');

    GF0x = keops_grad(F0,'x');
    g3= GF0x(x,y,b,p,a);
    
    % Compare with matlab
    g2 = zeros(3,Nx);
    dkernel_geomXY = - exp(-p*squmatrix_distance(x',y')) * p;
    for l=1:3 
        g2(l,:) = 2 * sum( a .* ( b * ( (repmat(x(l,:)',1,Ny)-repmat(y(l,:),Nx,1))  .* dkernel_geomXY)' ),1) ;
    end
    assert( allclose(g2,g3)==1 )
end


%-----------------------------------------------%
%   Test 4: fshape scp
%-----------------------------------------------%
function test_fshape_scp(testCase)
    nx = 15; d= 3; ny = 51;

    center_faceX = randn(nx,d); signalX = randn(nx,1); normalsX = 1+0*randn(nx,d);
    center_faceY = randn(ny,d); signalY = randn(ny,1); normalsY = randn(ny,d);

    kernel_size_geom = 1.23124;
    kernel_size_signal = 1.232;
    kernel_size_sphere =pi;

    opt.kernel_geom = 'cauchy';
    opt.kernel_signal = 'gaussian';
    opt.kernel_sphere = 'gaussian_oriented';

    res_cuda = shape_scp(center_faceX, center_faceY, signalX, signalY, normalsX, normalsY, kernel_size_geom, kernel_size_signal, kernel_size_sphere, opt);

    % Compare with matlab
    kernel_gaussian = @(r2,s) exp(-r2 / s^2);
    kernel_cauchy = @(r2,s)  1 ./ (1 + (r2/s^2));
    kernel_gaussian_oriented = @(prs,s)  exp( (-2 + 2*prs) / s^2);

    norm_normalsX = sqrt(sum(normalsX .^2,2));
    norm_normalsY = sqrt(sum(normalsY .^2,2));

    unit_normalsX = normalsX ./  repmat(norm_normalsX,1,size(normalsX,2));
    unit_normalsY = normalsY ./  repmat(norm_normalsY,1,size(normalsY,2));

    distance_signalXY = (repmat(signalX,1,ny)-repmat(signalY',nx,1)).^2;
    distance_center_faceXY=zeros(nx,ny);
    oriented_angle_normalsXY = zeros(nx,ny);

    for l=1:d
        distance_center_faceXY = distance_center_faceXY + (repmat(center_faceX(:,l),1,ny) - repmat(center_faceY(:,l)',nx,1)).^2;
        oriented_angle_normalsXY = oriented_angle_normalsXY + (repmat(unit_normalsX(:,l),1,ny).*repmat(unit_normalsY(:,l)',nx,1));
    end

    eval(['radial_function_geom=kernel_', lower(opt.kernel_geom), ';']);
    kernel_geomXY = radial_function_geom(distance_center_faceXY, kernel_size_geom);
    eval(['radial_function_signal=kernel_', lower(opt.kernel_signal), ';']);
    kernel_signalXY = radial_function_signal(distance_signalXY,kernel_size_signal);
    eval(['radial_function_sphere=kernel_', lower(opt.kernel_sphere) , ';']);
    kernel_sphereXY = radial_function_sphere(oriented_angle_normalsXY, kernel_size_sphere);

    res_matlab = sum(sum((norm_normalsX * norm_normalsY') .* kernel_geomXY .* kernel_signalXY .* kernel_sphereXY));

    assert( allclose(res_cuda,res_matlab)==1 )
end



%-----------------------------------------------%
%   Test 5: fshape scp dx
%-----------------------------------------------%
function test_fshape_scp_dx(testCase)
    nx = 15; d= 3; ny = 51;

    center_faceX = randn(nx,d); signalX = randn(nx,1); normalsX = 1+0*randn(nx,d);
    center_faceY = randn(ny,d); signalY = randn(ny,1); normalsY = randn(ny,d);

    kernel_size_geom = 1.23124;
    kernel_size_signal = 1.232;
    kernel_size_sphere =pi;

    opt.kernel_geom = 'cauchy';
    opt.kernel_signal = 'gaussian';
    opt.kernel_sphere = 'gaussian_oriented';

    res_cuda_dx = shape_scp_dx(center_faceX,center_faceY,signalX,signalY,normalsX,normalsY,kernel_size_geom,kernel_size_signal,kernel_size_sphere,opt);

    % Compare with matlab
    kernel_gaussian = @(r2,s) exp(-r2 / s^2);
    dkernel_cauchy = @(r2,s) -1 ./ (s^2 * (1 + (r2/s^2)) .^2);
    kernel_gaussian_oriented = @(prs,s)  exp( (-2 + 2*prs) / s^2);

    norm_normalsX = sqrt(sum(normalsX .^2,2));
    norm_normalsY = sqrt(sum(normalsY .^2,2));

    unit_normalsX = normalsX ./  repmat(norm_normalsX,1,size(normalsX,2));
    unit_normalsY = normalsY ./  repmat(norm_normalsY,1,size(normalsY,2));

    distance_signalXY = (repmat(signalX,1,ny)-repmat(signalY',nx,1)).^2;
    distance_center_faceXY=zeros(nx,ny);
    oriented_angle_normalsXY = zeros(nx,ny);

    for l=1:d
        distance_center_faceXY = distance_center_faceXY + (repmat(center_faceX(:,l),1,ny) - repmat(center_faceY(:,l)',nx,1)).^2;
        oriented_angle_normalsXY = oriented_angle_normalsXY + (repmat(unit_normalsX(:,l),1,ny).*repmat(unit_normalsY(:,l)',nx,1));
    end

    eval(['dradial_function_geom = dkernel_', lower(opt.kernel_geom), ';']);
    dkernel_geomXY = dradial_function_geom(distance_center_faceXY,kernel_size_geom);
    eval(['radial_function_signal = kernel_', lower(opt.kernel_signal), ';']);
    kernel_signalXY = radial_function_signal(distance_signalXY,kernel_size_signal);
    eval(['radial_function_sphere = kernel_', lower(opt.kernel_sphere) , ';']);
    kernel_sphereXY = radial_function_sphere(oriented_angle_normalsXY, kernel_size_sphere);

    for l=1:d
        res_matlab_dx(:,l) = 2* sum( (repmat(center_faceX(:,l),1,ny)-repmat(center_faceY(:,l)',nx,1)) .* (norm_normalsX * norm_normalsY') .* dkernel_geomXY .* kernel_signalXY .* kernel_sphereXY,2);
    end

    assert( allclose(res_cuda_dx,res_matlab_dx)==1 )
end
