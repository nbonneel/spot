all:
	g++ mainFIST.cpp UnbalancedSliced.cpp -O3 -fopenmp -mavx -o FIST -I.
	g++ mainColorTransfer.cpp UnbalancedSliced.cpp -O3 -fopenmp -mavx -o colorTransfer -I.
