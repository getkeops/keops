import os

class DebugConfig:

    # prints information about atomic operations during code building
    _debug_ops = False
    # adds C++ code for printing all input and output values for all atomic operations during computations
    _debug_ops_at_exec = False

    # Verbosity level (default is False unless KEOPS_VERBOSE define and not 0)
    _verbose = os.getenv("KEOPS_VERBOSE") != "0"

    def __init__(self):
        pass
    
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
    
    def set_verbose(self, verbose):
        self._verbose = verbose