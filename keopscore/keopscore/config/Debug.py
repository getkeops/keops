import os

from keopscore.utils.messages import KeOps_Warning


class DebugConfig:

    # prints information about atomic operations during code building
    _debug_ops = False
    # adds C++ code for printing all input and output values for all atomic operations during computations
    _debug_ops_at_exec = False

    def __init__(self):
        env_val = os.getenv("KEOPS_VERBOSE")
        if env_val is None:
            self._verbose = 1
        else:
            val = int(env_val)
            if val in (0, 1, 2):
                self._verbose = val
            else:
                KeOps_Warning(f"Invalid KEOPS_VERBOSE value: {env_val}. Verbose level must be 0, 1 or 2. Defaulting to 1.")
                self._verbose = 1

    def set_debug_ops(self, debug_ops):
        self._debug_ops = debug_ops

    def get_debug_ops(self):
        return self._debug_ops

    def set_debug_ops_at_exec(self, debug_ops_at_exec):
        self._debug_ops_at_exec = debug_ops_at_exec

    def get_debug_ops_at_exec(self):
        return self._debug_ops_at_exec

    def get_verbose(self):
        return self._verbose

    def set_verbose(self, val):
        if val in (0, 1, 2):
            self._verbose = val
        else:
            KeOps_Warning(f"Invalid verbose value: {val}. Verbose level must be 0, 1 or 2. Keeping previous value: {self._verbose}.")
            return
        
