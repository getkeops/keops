import pytest

from keopscore.formulas.GetReduction import GetReduction


@pytest.mark.parametrize("kw_name", ["lambda", "in", "from", "is", "global"])
def test_get_reduction_python_keyword_variable(kw_name):
    # Regression test for https://github.com/getkeops/keops/issues/426
    expected = GetReduction(
        "Sum_Reduction(Exp(sigma*SqNorm2(x-y))*beta,1)",
        aliases=[
            "x=Var(0,4,0)",
            "y=Var(1,4,1)",
            "beta=Var(2,4,1)",
            "sigma=Var(3,1,2)",
        ],
    )
    actual = GetReduction(
        f"Sum_Reduction(Exp({kw_name}*SqNorm2(x-y))*beta,1)",
        aliases=[
            "x=Var(0,4,0)",
            "y=Var(1,4,1)",
            "beta=Var(2,4,1)",
            f"{kw_name}=Var(3,1,2)",
        ],
    )
    assert repr(actual) == repr(expected)


def test_get_reduction_keyword_does_not_rewrite_prefix_identifier():
    expected = GetReduction(
        "Sum_Reduction((sigma + lambda_1)*SqNorm2(x-y),1)",
        aliases=[
            "x=Var(0,4,0)",
            "y=Var(1,4,1)",
            "sigma=Var(2,1,2)",
            "lambda_1=Var(3,1,2)",
        ],
    )
    actual = GetReduction(
        "Sum_Reduction((lambda + lambda_1)*SqNorm2(x-y),1)",
        aliases=[
            "x=Var(0,4,0)",
            "y=Var(1,4,1)",
            "lambda=Var(2,1,2)",
            "lambda_1=Var(3,1,2)",
        ],
    )
    assert repr(actual) == repr(expected)
