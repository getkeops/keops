function tests = generic_test
    tests = functiontests(localfunctions);
end


function setupOnce(testCase)  % do not change function name
    % set a new path, for example
    path_to_lib = '..';
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
%--------------------------------------%
%   Test 1: standard expression
%--------------------------------------%
function test_standard_expression(testCase)

    Nx = 50; Ny = 20;
    x = randn(3,Nx); y = randn(3,Ny); b = randn(3,Ny); u = randn(4,Nx); v = randn(4,Ny);
    p = .25;

    % Kernel with KeOps
    F = Kernel('x=Vx(3)','y=Vy(3)','u=Vx(4)','v=Vy(4)','b=Vy(3)', 'p=Pm(1)', 'Square((u,v))*Exp(-p*SqNorm2(x-y))*b');
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
    F1 = Kernel('x=Vx(0,3)','y=Vy(1,3)','b=Vy(2,3)','a=Vx(3,3)', 'p=Pm(4,1)', 'Grad(Exp(-p*SqNorm2(x-y))*b,x,a)');
    g1 = F1(x,y,b,a,p);
    
    % Compare with matlab
    g2 = zeros(3,Nx);
    dKernel_geomXY = - exp(-p*squmatrix_distance(x',y')) * p;
    for l=1:3 
        g2(l,:) = 2 * sum( a .* ( b * ( (repmat(x(l,:)',1,Ny)-repmat(y(l,:),Nx,1))  .* dKernel_geomXY)' ),1) ;
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

    F0 = Kernel('x=Vx(0,3)','y=Vy(1,3)','b=Vy(2,3)', 'p=Pm(3,1)', 'Exp(-p*SqNorm2(x-y))*b');

    GF0x = GradKernel(F0,'x','e=Vx(4,3)');
    g3= GF0x(x,y,b,p,a);
    
    % Compare with matlab
    g2 = zeros(3,Nx);
    dKernel_geomXY = - exp(-p*squmatrix_distance(x',y')) * p;
    for l=1:3 
        g2(l,:) = 2 * sum( a .* ( b * ( (repmat(x(l,:)',1,Ny)-repmat(y(l,:),Nx,1))  .* dKernel_geomXY)' ),1) ;
    end
    assert( allclose(g2,g3)==1 )
end



