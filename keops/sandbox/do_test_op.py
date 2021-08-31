import sys
from keops.test.TestOperation import TestOperation

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("you should provide an operation to test, e.g. 'python do_test_op.py Exp'")
    else:
        res = TestOperation(sys.argv[1])