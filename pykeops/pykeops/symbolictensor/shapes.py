################################################
###  rules for checking and infering shapes
################################################

from math import prod

class Shape(tuple):
    def broadcast(*shapes):
        # implements the broadcasting rule for shapes :
        # we check that their shapes are compatible for broadcasting
        # and return the output broadcasted shape.
        # for example if shapes[0]=(2,3,1) and shapes[1]=(2,1,4), then
        # x.broadcast(y) will return (2,3,4)
        shapes = list(shapes)
        ndims = list(len(shape) for shape in shapes)
        ndim = max(ndims)
        for i in range(len(shapes)):
            shapes[i] = shapes[i] + (1,) * (ndim - ndims[i])
        shapeout = []
        for k in range(ndim):
            dims = set(shape[k] for shape in shapes)
            if len(dims) == 2 and min(dims) == 1:
                dimout = max(dims)
            elif len(dims) == 1:
                dimout = dims.pop()
            else:
                raise ValueError(
                    f"Incompatible shapes for broadcasting. The axis dimensions at non-singleton dimension {k} are {', '.join(list(str(x.shape[k]) for x in args))}."
                )
            shapeout.append(dimout)
        return Shape(shapeout)
    
    def reduce(self, axis=None, keepdim=False):
        # implements reduction rule for shape, corresponding to e.g. "sum" operation in NumPy or PyTorch
        # ex : reduce((4,5,3), axis=1) returns (4,3)
        # ex : reduce((4,5,3), axis=(0,1)) returns (3,)
        # ex : reduce((4,5,3), axis=1, keepdim=True) returns (4,1,3)
        # ex : reduce((4,5,3), axis=None) returns ()
        if axis is None:
            axis = tuple(range(len(self)))
        if isinstance(axis,int):
            axis = [axis]
        shapeout = list(self)
        dec = 0
        for k in range(len(axis)):
            if keepdim:
                shapeout[k] = 1
            else:
                shapeout.pop(k-dec)
                dec += 1
        return Shape(shapeout)


class BroadcastShapes:
    @staticmethod
    def get_shape(*args):
        nargs = len(args)
        shapes = [arg.shape for arg in args]
        return Shape.broadcast(*shapes)

    @staticmethod
    def test_non_trivial_inner_broadcast(args):
        set_shapes = set(arg.inner_shape for arg in args)
        n_sh = len(set_shapes)
        if n_sh > 2:
            return True
        elif n_sh == 2:
            dims = [prod(shape) for shape in set_shapes]
            return min(dims) != 1
        else:
            return False


class ReductionShape:
    @staticmethod
    def get_shape(arg, axis=None, keepdim=False):
        # implements the reduction rule for shapes.
        return Shape.reduce(arg.shape, axis=axis, keepdim=keepdim)
