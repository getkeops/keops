class F(torch.autograd.Function):
    @staticmethod
    def forward(ctx, linop, b, *args):
        ctx.linop = linop
        ctx.args = args
        a = ConjugateGradientSolver(linop,b.data,eps=1e-6)
        ctx.save_for_backward(a)
        return a
    @staticmethod
    def backward(ctx, grad_output):
        linop = ctx.linop
        args = ctx.args
        a, = ctx.saved_tensors
        e = InvLinOp(linop,grad_output)
        with torch.enable_grad():
            linop(a.data).backward(-e)
        return (None, e) + tuple(map(lambda x : x.grad,args))
