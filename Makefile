all:
	g++ FIST.cpp UnbalancedSliced.cpp -O3 -fopenmp -mavx -o FIST
