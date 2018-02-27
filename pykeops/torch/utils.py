import torch

def _squared_distances(x, y) :
    x_i = x.unsqueeze(1)         # Shape (N,D) -> Shape (N,1,D)
    y_j = y.unsqueeze(0)         # Shape (M,D) -> Shape (1,M,D)
    return ((x_i-y_j)**2).sum(2) # N-by-M matrix, xmy[i,j] = |x_i-y_j|^2

def _scalar_products(u, v) :
    u_i = u.unsqueeze(1)         # Shape (N,D) -> Shape (N,1,D)
    v_j = v.unsqueeze(0)         # Shape (M,D) -> Shape (1,M,D)
    return (u_i*v_j).sum(2) # N-by-M matrix, usv[i,j] = <u_i,v_j>

def _log_sum_exp(mat, dim):
    """
    Computes the log-sum-exp of a matrix with a numerically stable scheme, 
    in the user-defined summation dimension: exp is never applied
    to a number >= 0, and in each summation row, there is at least
    one "exp(0)" to stabilize the sum.
    
    For instance, if dim = 1 and mat is a 2d array, we output
                log( sum_j exp( mat[i,j] )) 
    by factoring out the row-wise maximas.
    """
    max_rc = torch.max(mat, dim)[0]
    return max_rc + torch.log(torch.sum(torch.exp(mat - max_rc.unsqueeze(dim)), dim))




class Formula :
    def __init__(self, formula_sum=None, routine_sum=None, 
                       formula_log=None, routine_log=None, intvalue = None):
        if intvalue is None :
            self.formula_sum = formula_sum
            self.routine_sum = routine_sum
            self.formula_log = formula_log
            self.routine_log = routine_log
        else :
            self.formula_sum =  "IntCst("+str(intvalue)+")"
            self.routine_sum = lambda **x :   intvalue
            self.formula_log =  "Log(IntCst("+str(intvalue)+")"
            self.routine_log = lambda **x : math.log(intvalue)
            self.intvalue    = intvalue
    
    def __add__(self, other) :
        return Formula( formula_sum =           "("+self.formula_sum  + " + " + other.formula_sum +")",
                        routine_sum = lambda **x :  self.routine_sum(**x) +     other.routine_sum(**x) ,
                        formula_log =      "Log( ("+self.formula_sum  + " + " + other.formula_sum+") )",
                        routine_log = lambda **x : (self.routine_sum(**x) +     other.routine_sum(**x)).log() ,
                )

    def __mul__(self, other) :
        return Formula( formula_sum =           "("+self.formula_sum  + " * " + other.formula_sum + ")" ,
                        routine_sum = lambda **x :  self.routine_sum(**x) *     other.routine_sum(**x) ,
                        formula_log =           "("+self.formula_log  + " + " + other.formula_log + ")",
                        routine_log = lambda **x : (self.routine_log(**x) +     other.routine_log(**x)) ,
                )

    def __pow__(self, other) :
        "N.B.: other should be a Formula(intvalue=N)."
        return Formula( formula_sum =        "Pow("+self.formula_sum+   ","+str(other.intvalue)+")" ,
                        routine_sum = lambda **x :  self.routine_sum(**x)**(    other.intvalue ),
                        formula_log =          "("+other.formula_sum+   " * "+self.formula_log + ")",
                        routine_log = lambda **x : other.routine_sum(**x) *   self.routine_log(**x) ,
                )

