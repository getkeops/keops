import pykeops.config as pykeopsconfig


class _KeOpsBinderMap:
    """Lazily load cpp/nvrtc binders on first use to avoid eager CUDA init (#444)."""

    def __init__(self):
        self._binders = None

    def _get_binders(self):
        if self._binders is None:
            from .cpp import LoadKeOps_cpp

            if pykeopsconfig.cuda.get_use_cuda():
                from .nvrtc import LoadKeOps_nvrtc

                self._binders = {
                    "nvrtc": LoadKeOps_nvrtc.LoadKeOps_nvrtc,
                    "cpp": LoadKeOps_cpp.LoadKeOps_cpp,
                }
            else:
                self._binders = {"cpp": LoadKeOps_cpp.LoadKeOps_cpp}
        return self._binders

    def __getitem__(self, key):
        return self._get_binders()[key]

    def __iter__(self):
        return iter(self._get_binders())

    def __len__(self):
        return len(self._get_binders())

    def __contains__(self, key):
        return key in self._get_binders()

    def keys(self):
        return self._get_binders().keys()

    def values(self):
        return self._get_binders().values()

    def items(self):
        return self._get_binders().items()


keops_binder = _KeOpsBinderMap()
