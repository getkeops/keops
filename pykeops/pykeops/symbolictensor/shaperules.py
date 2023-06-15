################################################
###  rules for checking and infering shapes
################################################

from math import prod

class BroadcastShapes:
    @staticmethod
    def get_shape(args, params=[]):
        # implements the broadcasting rule for shapes :
        # given a list of arguments x,y,z,... representing tensors
        # we check that their shapes are compatible for broadcasting
        # and return the output broadcasted shape.
        # for example if x.shape=(2,3,1) and y.shape=(2,1,4), then
        # get_shape([x,y]) will return (2,3,4)
        # N.B. input params is unused here.
        nargs = len(args)
        shapes = [arg.shape for arg in args]
        ndims = list(len(shape) for shape in shapes)
        ndim = max(ndims)
        for i in range(nargs):
            shapes[i] = shapes[i] + (1,)*(ndim-ndims[i])
        shapeout = []
        for k in range(ndim):
            dims = set(shape[k] for shape in shapes)
            if len(dims) == 2 and min(dims) == 1:
                dimout = max(dims)
            elif len(dims) == 1:
                dimout = dims.pop()
            else:
                raise ValueError(f"Incompatible shapes for broadcasting. The axis dimensions at non-singleton dimension {k} are {', '.join(list(str(x.shape[k]) for x in args))}.")
            shapeout.append(dimout)
        return tuple(shapeout)
    
    @staticmethod
    def test_non_trivial_inner_broadcast(args):
        set_shapes = set(arg.inner_shape for arg in args)
        n_sh = len(set_shapes)
        if n_sh > 2:
            return True
        elif n_sh == 2:
            dims = [prod(shape) for shape in set_shapes]
            return min(dims)!=1
        else:
            return False
        
        


class ReductionShape:
    @staticmethod
    def get_shape(args, params):
        # implements the reduction rule for shapes.
        (arg,) = args  # there should be only one argument by default
        axis, keepdim = params  # TODO change this...
        shapeout = list(arg.shape)
        if keepdim:
            shapeout[axis] = 1
        else:
            shapeout.pop(axis)
        return tuple(shapeout)
