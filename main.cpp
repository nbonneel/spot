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
