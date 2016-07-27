BINDIR = ./
EXECUTABLE := testCPU
	
all: testCPU testCUDA

testCPU: wallsolverCPU.cu
	nvcc wallsolverCPU.cu -o testCPU

testCUDA: board.cu solver.cu
	nvcc board.cu solver.cu -o testCUDA