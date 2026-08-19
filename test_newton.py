import pytest
import numpy as np
import math

import newton


def test_basic_function():
    assert np.isclose(newton.optimize(np.cos, 2.95), math.pi, atol=1e-3)

square = lambda x: x**2
    
def test_basic_function_2():
    def test_fun(x):
        return(x**2)
    assert np.isclose(newton.optimize(test_fun, 1), 0, atol=1e-3)

def test_basic_function_2_different_starting_point():
    def test_fun(x):
        return(x**2)
    assert np.isclose(newton.optimize(test_fun, 100), 0, atol=1e-3)


def test_bad_input():
    ## Ideally, our function would raise the exception with a useful message.
    with pytest.raises(TypeError, match='`x0` must be numeric'):
        newton.optimize(2.95, np.cos)

## How to check that a warning is (correctly) emitted:
## def test_warning():
##    with pytest.warns(UserWarning, match='greater'):
##        newton.optimize(...., ....)