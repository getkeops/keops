class ChunksConfig:
    """
    Configuration and state management for chunked computation schemes. 
    Special computation scheme for dim>100
    """
    
    # Constants for chunking strategy
    _dimchunk = 64
    _dim_treshold_chunk = 146
    _specdims_use_chunk = [99, 100, 102, 120, 133, 138, 139, 140, 141, 142]
    
    _enable_chunks = True
    _dimfinalchunk = 64
    _enable_final_chunk = True
    _mult_var_highdim = False
    
    def __init__(self):
        pass
    
    def get_dimchunk(self):
        """Get the current chunk dimension."""
        return self._dimchunk

    def get_dim_treshold_chunk(self):
        """Get the current dimension threshold for chunking."""
        return self._dim_treshold_chunk
    
    def get_specdims_use_chunk(self):
        """Get the current list of specific dimensions for which chunking is used."""
        return self._specdims_use_chunk
    
    def get_enable_chunks(self):
        """Get the current enable_chunk state."""
        return self._enable_chunks
    
    def set_enable_chunks(self, val):
        """Set enable_chunk state from int (1=True, 0=False, -1=no change)."""
        if val == 1:
            self._enable_chunks = True
        elif val == 0:
            self._enable_chunks = False
        # val == -1 means keep previous value
    
    def get_dimfinalchunk(self):
        """Get the current final chunk dimension."""
        return self._dimfinalchunk
    
    def set_dimfinalchunk(self, val):
        """Set the final chunk dimension."""
        self._dimfinalchunk = val
    
    def set_enable_finalchunk(self, val):
        """Set enable_final_chunk state from int (1=True, 0=False, -1=no change)."""
        if val == 1:
            self._enable_final_chunk = True
        elif val == 0:
            self._enable_final_chunk = False
        # val == -1 means keep previous value

    def get_enable_final_chunk(self):
        """Get the current enable_final_chunk state."""
        return self._enable_final_chunk
    
    def set_mult_var_highdim(self, val):
        """Set mult_var_highdim state from int (1=True, 0=False, -1=no change)."""
        if val == 1:
            self._mult_var_highdim = True
        elif val == 0:
            self._mult_var_highdim = False
        # val == -1 means keep previous value

    def get_mult_var_highdim(self):
        """Get the current mult_var_highdim state."""
        return self._mult_var_highdim
            
    def use_final_chunks(self, red_formula):
        """Determine if final chunks mode should be used for this formula."""
        return (
            self.get_enable_final_chunk()
            and self.get_mult_var_highdim() 
            and red_formula.dim > self.get_dim_treshold_chunk()
        )
