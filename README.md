# SPOT: Sliced Partial Optimal Transport

 ### Nicolas Bonneel and David Coeurjolly
 ### Univ. Lyon, CNRS

## Demonstration code.

See `mainFIST.cpp`, this example demonstrates the fast iterative sliced transport (**fist**) of
two random pointsets in dimension three. The `FIST` executable outputs the transformation
(translation, rotation and scaling) to apply to the first point set to match (in the sense of the sliced optimal transport) the second one.

`mainColorTransfer.cpp` is an example of color transfer between an image and a larger one (see `Datasets/Images/`). The result is given in `outtransfer.png`.

## Build

* Linux:

        make  
        ./FIST
        ./colorTransfer

* MacOS (require external OpenMP when using Apple Clang, *e.g.* `brew install libomp`):

        make -f Makefile.osx
        ./FIST
        ./colorTransfer

* Windows: use the Visual Studio Project.


## Datasets

We provide some data (pointsets, images) that were used for the SIGGRAPH Paper:

* 3D pointsets (`mumble` and `castle` objects). Note that partial (with suffices `cut`) and global pointsets are centered. To reproduce a FIST test, you may need to draw a random rotation and scaling and apply it to one of the pointsets.
* Images for the partial color transfer application
* (2D coming soon)

## Citation

``` bibtex
@article{bonneel19SPOT,
    author = "Bonneel, Nicolas and Coeurjolly, David",
    title = "Sliced Partial Optimal Transport",
    journal = "{ACM} Transactions on Graphics (Proceedings of SIGGRAPH)",
    year = "2019",
    abstract = "Optimal transport research has surged in the last decade with wide applications in computer graphics. In most cases, however, it has focused on the special case of the so-called “balanced” optimal transport problem, that is, the problem of optimally matching positive measures of equal total mass. While this approach is suitable for handling probability distributions as their total mass is always equal to one, it precludes other applications manipulating disparate measures. Our paper proposes a fast approach to the optimal transport of constant distributions supported on point sets of different cardinality via one-dimensional slices. This leads to one-dimensional partial assignment problems akin to alignment problems encountered in genomics or text comparison. Contrary to one-dimensional balanced optimal transport that leads to a trivial linear-time algorithm, such partial optimal transport, even in 1-d, has not seen any closed-form solution nor very efficient algorithms to date. We provide the first efficient 1-d partial optimal transport solver. Along with a quasilinear time problem decomposition algorithm, it solves 1-d assignment problems consisting of up to millions of Dirac distributions within fractions of a second in parallel. We handle higher dimensional problems via a slicing approach, and further extend the popular iterative closest point algorithm using optimal transport – an algorithm we call Fast Iterative Sliced Transport. We illustrate our method on computer graphics applications such a color transfer and point cloud registration.",
    volume = "38",
    number = "4",
    month = "jul"
}
```

## License

```
  Copyright (c) 2019 CNRS
  Nicolas Bonneel <nicolas.bonneel@liris.cnrs.fr>
  David Coeurjolly <david.coeurjolly@liris.cnrs.fr>

  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

  1. Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIEDi
  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
  ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
