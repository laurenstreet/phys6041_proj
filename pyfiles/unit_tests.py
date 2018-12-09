##### Phys 6041 Project - University of Cincinnati:  Fits for Dark Matter Direct Detection
##### Constants

##### import necessary modules
import numpy as np

def assertfloat(x):
    '''Test that a data type is a float.

    This produces a test that x is an numpy.float64 object.

    Parameters
    ----------
    x : any
        Parameter to test.

    Returns
    -------
    no output
        Will not return anything if x is a float.
    ValueError
        Will return "x is not a numpy.float64" if x is not a float
    '''
    if type(x) != np.float64:
        raise ValueError(f"{x} is not a numpy.float64")
    assert(isinstance(x,np.float))
    return

def assertarray(x):
    '''Test that a data type is an array.

    This produces a test that x is an numpy.ndarray object.

    Parameters
    ----------
    x : any
        Parameter to test.

    Returns
    -------
    no output
        Will not return anything if x is an array.
    ValueError
        Will return "x is not a numpy.ndarray" if x is not an array
    '''
    if type(x) != np.ndarray:
        raise ValueError(f"{x} is not a numpy.ndarray")
    assert(isinstance(x,np.ndarray))
    return

def assertshape(x,y):
    '''Test the shape of an input.

    This produces a test that the shape of x is y.

    Parameters
    ----------
    x : any
        Parameter to test.
    y : list (N,M)
        Tested shape for x.

    Returns
    -------
    no output
        Will not return anything if shape of x is y.
    ValueError
        Will return "numpy.shape(x) not to y" if shape of x is not y.
    '''
    if np.shape(x) != y:
        raise ValueError(f"numpy.shape({x}) not equal to {y}")
    assert(np.shape(x) == y)
    return