function out = EvalFormula(F,varargin)
fname = ['F',encodestring(F)];
if ~(exist(fname,'file')==3)
    buildFormula(F)
end
out = feval(fname,varargin{:});
