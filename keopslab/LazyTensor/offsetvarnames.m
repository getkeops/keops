
function newformula = offsetvarnames(formula,nvars,offset)
newformula = formula;
for k=nvars:-1:1
    newformula = strrep(newformula,['v',num2str(k)],...
        ['v',num2str(k+offset)]);
end
