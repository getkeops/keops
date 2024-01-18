import sys
from keopscore.utils.TestFormula import TestFormula

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(
            """
            you should provide a formula to test, 
            e.g. 'python do_test_formula.py "Exp(-Sum((Var(0,3,0)-Var(1,3,1))**2))*Var(2,1,1)"'
            """
        )
    else:
        if len(sys.argv) == 2:
            res = TestFormula(sys.argv[1])
        elif len(sys.argv) == 3:
            res = TestFormula(sys.argv[1], dtype=sys.argv[2])
        else:
            res = TestFormula(
                sys.argv[1], dtype=sys.argv[2], test_grad=eval(sys.argv[3])
            )
