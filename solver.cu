/*
	

*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include "board.cu"








__global__ void
solveForAllWalls() {
	int tidx = threadIdx.x;	// X-Dim = Wall
	int tidy = threadIdx.y;	// Y-Dim = Direction
	int space = blockIdx.x;	// Space #

	// 


}




/* Kernel

*/
__global__ void
calcBestMove() {
	int gtid, block, tidx, tidy;
	tidx = threadIdx.x;
	tidy = threadIdx.y;
	block = blockIdx.x;

	gtid = block * blockDim.x + tidx + tidy;
	
	// Block = Space
	// x = tid.x 	wall
	// y = tid.y	wall direction 0 - 4

	// moveWall (walls, tid, newDir, position)

	CUDA_moveWallParallel2D(walls,)

}




// CUDA Error Check
void checkCudaError(cudaError_t e, char in[]) {
	if (e != cudaSuccess) {
		printf("CUDA Error: %s, %s \n", in, cudaGetErrorString(e));
		exit(EXIT_FAILURE);
	}
}


int main(int argc, char const *argv[])
{
	
	int numSpaces = SPACE_LENGTH * SPACE_WIDTH;
	size_t spaceSize = sizeof(space) * numSpaces;

	int numWalls = WALL_LENGTH * WALL_WIDTH;
	size_t wallSize = sizeof(wall) * numWalls;

	// Malloc the array of wall / board
	wall *walls = (wall *)malloc(wallSize);
	space *board = (space *)malloc(spaceSize);


	// Initialize, zero out the board 
	boardInit(board);
	// Generate walls 
	generateWalls(walls);
	generateBoard(board, walls);



	// Malloc space on device, copy to device
	walls *d_walls = NULL;
	space *d_board = NULL;

	checkCudaError( cudaMalloc((void**) &d_walls, wallSize), 
		"Malloc Histogram");
	checkCudaError( cudaMalloc((void**) &d_board, spaceSize), 
		"Malloc Atom List");

	checkCudaError( cudaMemcpy(d_histogram, histogram, histogramSize, cudaMemcpyHostToDevice), 
		"Copy histogram to Device");
	checkCudaError( cudaMemcpy(d_atom_list, atom_list, atomSize, cudaMemcpyHostToDevice), 
		"Copy atom_list to Device");




	// Setup: Measure Runtime
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	// CUDA Kernel Call
	//PDH_baseline <<<ceil(PDH_acnt/32), 32>>> (d_histogram, d_atom_list, PDH_res, PDH_acnt);

	checkCudaError(cudaGetLastError(), "Checking Last Error, Kernel Launch");

	// Report kernel runtime
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Time to generate: %f ms \n", elapsedTime);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);




	return 0;
}