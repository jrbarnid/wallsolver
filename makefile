BINDIR = ./
EXECUTABLE := testCPU
	
all: testCPU testCUDA

testCPU: wallsolverCPU.cu
	nvcc wallsolverCPU.cu -o testCPU

testCUDA: boardCPU.cu board.cu solver.cu
	nvcc boardCPU.cu board.cu solver.cu -o testCUDA