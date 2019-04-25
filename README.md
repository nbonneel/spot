# SPOT: Sliced Partial Optimal Transport

 ### Nicolas Bonneel and David Coeurjolly
 ### Univ. Lyon, CNRS

Demonstration code.

See `main.cpp`, this example demonstrates the fast iterative sliced transport (**fist**) of
two random pointsets in dimension three. The SPOT executable outputs the transformation
(translation, rotation and scaling) to apply to the first point set to match (in the sense of the sliced optimal transport) the second one.

## Build

* Linux:

        make  
        ./SPOT

* MacOS (require external OpenMP when using Apple Clang, *e.g.* `brew install libomp`):

        make -f Makefile.osx
        ./SPOT

* Windows: use the Visual Studio Project.
