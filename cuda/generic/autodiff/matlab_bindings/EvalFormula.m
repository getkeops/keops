function out = EvalFormula(F,varargin)
fname = [encodestring(F),'.',mexext];
if ~(exist(fname,'file')==3)
    buildFormula(F)
end
eval(['!mv "build/',fname,'" build/tmp.',mexext])
out = tmp(varargin{:});
eval(['!mv build/tmp.',mexext,' "build/',fname,'"'])


