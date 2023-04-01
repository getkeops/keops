from pykeops.numpy import Genred


class GenredR(Genred):
    r"""Extension of the Genred class for RKeOps. Just an empty shell
    that passes arguments to :class:`Genred`."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, list_args, *args, **kwargs):
        return super().__call__(*list_args, *args, **kwargs)
