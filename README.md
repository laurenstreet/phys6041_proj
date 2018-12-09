# Phys 6041 Project - University of Cincinnati:  Fits for Dark Matter Direct Detection

With this code one can calculate various functions relevant for the kinematics involved in direct detection scattering processes.  The user must supply their own differential scattering cross-section or use the one that I hardcoded into the project files which represents the simplified model explained in the included file proj_summary.pdf.  The differential cross-section in the code was obtained using the Mathematica packages DirectDM and DMFormfactor (see acknowledgements for references).

There are two main pipelines that one can follow to obtain best-fit values for Wilson coefficents.  One main pipeline is hardcoded for the model explained in the summary pdf for the Xenon100 experiment and many of the relevant functions only require the recoil energy and DM mass as an input.  The other pipeline consists of the same functions, with the exception of plot functions, generalized so that the user has more freedom in terms of models and experiments to be analyzed.  For information on the Xenon100 experiment see acknowledgements.


## Getting Started

To run these functions one must import the proper packages.  To import relevant constants,

```
import pyfiles.proj_constants
```

to import the general pipeline,
```
import pyfiles.proj_functions
```

and to import the pipeline hardoced for the Xenon100 experiment,
```
import pyfiles.proj_functions_xe100
```


### Prerequisites

In order to run all functions one must have the python package Numpy while plot functions require Matplotlib and harcoded Xenon100 functions require Numba.

All package files require the included file xenon100_detector_table.dat.


## Running the tests

There are tests for each package given as Jupyter notebooks.  For a given input, one can test its shape or its data type (only float or array).


## Running the examples

There are examples for each package given as Jupyter notebooks.


## Authors

[laurenstreet](https://github.com/laurenstreet)


## Acknowledgments
DMFormfactor:  A.L. Fitzpatrick, W. Haxton, E. Katz, N. Lubbers, and Y. Xu (2012) e-print: 1203.3542.

DMFormfactor:  N. Anand, A. L. Fitzpatrick, and W.C. Haxton (2013) e-print:  1308.6288.

DirectDM:  F. Bishara, J. Brod, B. Grinstein, and J. Zupan (2017) e-print:  1708.02678.

Xenon100:  XENON collaboration and B. Farmer (2017) e-print:  1705.02614.

Thanks to Joachim Brod, Henry Schreiner, Michele Tammaro, L.C.R. Wijewardhana, and Jure Zupan for valuable discussions and to the Physics department at the University of Cincinnati for financial support in the form of the Violet M. Diller Fellowship.
