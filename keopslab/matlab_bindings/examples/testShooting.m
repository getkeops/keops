function testShooting

addpath('../build')
addpath('..')

q0 = rand(2,15)*2-1;
p0 = 3*randn(size(q0));

sigma = .5;
oos2 = 1/sigma^2;
[d,n] = size(q0);

K = Kernel('Vx(1,2)','Vy(2,2)','DivFreeGaussKernel(2)');
%K = Kernel('Vx(1,2)','Vy(2,2)','CurlFreeGaussKernel(2)');
%K = Kernel('Vx(1,2)','Vy(2,2)','GaussKernel_(2,2)');

GK = GradKernel(K,'Vx(1,2)','Vx(4,2)');

ng = 50;
[x1,x2] = ndgrid(linspace(-1,1,ng));
x = [x1(:)';x2(:)'];
[~,PQX] = ode45(@FlowEq,linspace(0,1,100),join(p0,q0,x));
[~,Q,X] = split(PQX');
clf
hold on
axis equal
axis off
h = [];
h1 = [];
h2 = [];
ax = [min(min(X(1,:,:))),max(max(X(1,:,:))),...
    min(min(X(2,:,:))),max(max(X(2,:,:)))];
while 1
for k=2:size(X,3)
    x1 = reshape(X(1,:,k),ng,ng);
    x2 = reshape(X(2,:,k),ng,ng);
    delete(h)
    h = plot(squeeze(Q(1,:,1:k))',squeeze(Q(2,:,1:k))','r','Linewidth',3);
    delete(h1)
    h1 = plot(x1,x2,'b');
    delete(h2)
    h2 = plot(x1',x2','b');
    axis(ax)
    shg
end
end

    function dotpq = GeodEq(t,pq)
        [p,q] = split(pq);
        dotpq = join(-GK(oos2,q,q,p,p),K(oos2,q,q,p));
    end

    function dotpqx = FlowEq(t,pqx)
        [p,q,x] = split(pqx);
        dotpqx = join(-GK(oos2,q,q,p,p),K(oos2,q,q,p),K(oos2,x,q,p));
    end

    function [p,q,x] = split(pqx)
        nt = size(pqx,2);
        nx = size(pqx,1)/d-2*n;
        p = reshape(pqx(1:n*d,:),d,n,nt);
        q = reshape(pqx(n*d+1:2*n*d,:),d,n,nt);
        x = reshape(pqx(2*n*d+1:end,:),d,nx,nt);
    end

    function pqx = join(p,q,x)
        if nargin==2
            x = [];
        end
        nt = size(p,3);
        nx = size(x,2);
        pqx = [reshape(p,n*d,nt);reshape(q,n*d,nt);reshape(x,nx*d,nt)];
    end

end