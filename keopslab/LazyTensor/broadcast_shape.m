function newshape = broadcast_shape(sx,sy)
sx = [sx,ones(1,length(sy)-length(sx))];
sy = [sy,ones(1,length(sx)-length(sy))];
newshape = max(sx,sy);
% now we check compatibility
tmp = min(sx,sy);
if ~all(tmp==1 | tmp==newshape)
    error('incompatible dimensions')
end

