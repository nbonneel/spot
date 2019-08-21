/*
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
*/
#include "UnbalancedSliced.h"


int main()
{
	omp_set_nested(0);

	int M = 700, N = 1000;
	int FIST_iters = 200;
	int slices = 100;
	UnbalancedSliced sliced;

	std::vector<Point<3, double> > randomPoint1(M);
	std::vector<Point<3, double> > randomPoint2(N);
	
	for (int i=0; i<M; i++) {
		randomPoint1[i][0] = rand()/(double)RAND_MAX;
		randomPoint1[i][1] = rand()/(double)RAND_MAX;
		randomPoint1[i][2] = rand()/(double)RAND_MAX;
	}
	for (int i=0; i<N; i++) {
		randomPoint2[i][0] = rand()/(double)RAND_MAX * 2.0 + 2;
		randomPoint2[i][1] = rand()/(double)RAND_MAX * 2.0 + 4;
		randomPoint2[i][2] = rand()/(double)RAND_MAX * 2.0 + 6; 
	}	
	std::vector<double> rot(9);
	std::vector<double> trans(3);
	double scaling;
	sliced.fast_iterative_sliced_transport(FIST_iters, slices, randomPoint1, randomPoint2, rot, trans, true, scaling);

	std::cout << "scale: " << scaling << std::endl;
	std::cout << "translation: " << trans[0] << ", " << trans[1] << ", " << trans[2] << std::endl;
	std::cout << "rotation: " << std::endl;
	std::cout << rot[0] << " " << rot[1] << " " << rot[2] << std::endl;
	std::cout << rot[3] << " " << rot[4] << " " << rot[5] << std::endl;
	std::cout << rot[6] << " " << rot[7] << " " << rot[8] << std::endl;
	std::cout << std::endl;

	return 0;
}
