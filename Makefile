all:
	g++ mainFIST.cpp UnbalancedSliced.cpp -O3 -fopenmp -mavx -o FIST
