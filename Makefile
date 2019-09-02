all:
	g++ mainFIST.cpp UnbalancedSliced.cpp -O3 -fopenmp -mavx -o FIST -I. --std=c++11
	g++ mainColorTransfer.cpp UnbalancedSliced.cpp -O3 -fopenmp -mavx -o colorTransfer -I.  --std=c++11
