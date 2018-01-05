classdef KernelClass < handle
    properties
        Fname
        indxy
    end
    methods
        function obj = KernelClass(varargin)
            obj.indxy = [-1,-1];
            formula = varargin{nargin};
            fname = formula;
            CodeFormula = ['#define F ',formula];
            CodeVars = '';
            for k=1:nargin-1
                str = varargin{k};
                fname = [str,';',fname];
                [varname,vartype] = sepeqstr(str);
                if ~isempty(varname)
                    CodeVars = [CodeVars,'decltype(',vartype,') ',varname,';'];
                end
                if vartype(1)=='x' && obj.indxy(1)==-1
                    obj.indxy(1) = str2num(vartype(2))+1;
                elseif vartype(1)=='y' && obj.indxy(2)==-1
                    obj.indxy(2) = str2num(vartype(2))+1;
                end
            end
            filename = [fname,'.',mexext];
            if ~(exist(filename,'file')==3)
                buildFormula(CodeVars,CodeFormula,filename)
            end
            obj.Fname = ['F',clockstr,num2str(floor(rand(1)*1e16))];
            eval(['!cp "build/',filename,'" build/',obj.Fname,'.',mexext])
        end
        function out = Eval(obj,varargin)
            nx = size(varargin{obj.indxy(1)},2);
            ny = size(varargin{obj.indxy(2)},2);
            out = feval(obj.Fname,nx,ny,varargin{:});
        end
        function delete(obj)
            eval(['!rm build/',obj.Fname,'.',mexext])
        end
    end
end

function buildFormula(code1,code2,filename)
mysetenv('PATH','/Developer/NVIDIA/CUDA-9.1/bin/')
mysetenv('PATH','/Applications/MATLAB_R2014a.app/bin/')
cd ..
eval(['!./compile_mex_cpu "',code1,'" "',code2,'"'])
cd matlab_bindings
eval(['!mv build/tmp.',mexext,' "build/',filename,'"'])
end

function mysetenv(var,string)
if isempty(strfind(getenv(var),string))
    setenv(var, [getenv(var) ':' string])
end
end

function c = clockstr
c = num2str(clock');
c(:,end+1) = '_';
c = c';
c = strrep(c(:)',' ','');
c = strrep(c,'.','_');
end

function [left,right] = sepeqstr(str)
% get string before and after equal sign
pos = find(str=='=');
if isempty(pos)
    pos = 0;
end
left = str(1:pos-1);
right = str(pos+1:end);
end



