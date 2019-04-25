all:
	g++ main.cpp UnbalancedSliced.cpp -O3 -fopenmp -mavx -o FIST
