BINDIR = ./
EXECUTABLE := testCPU
	
testCPU: wallsolverCPU.cu
	nvcc wallsolverCPU.cu -o testCPU