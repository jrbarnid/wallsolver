BINDIR = ./
EXECUTABLE := testCPU
	
all: testCPU testCUDA

testCPU: wallsolverCPU.cu
	nvcc wallsolverCPU.cu -o testCPU

testCUDA: board.cu boardCPU.cu solver.cu 
	nvcc board.cu boardCPU.cu solver.cu -o testCUDA