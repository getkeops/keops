function F = Kernel(varargin)
    obj = KernelClass(varargin{:});
    F = @obj.Eval;
end