BINDIR = ./
EXECUTABLE := testCPU
	
all: testCPU testCUDA

testCPU: wallsolverCPU.cu
	nvcc wallsolverCPU.cu -o testCPU

testCUDA: solver.cu
	nvcc solver.cu -o testCUDA