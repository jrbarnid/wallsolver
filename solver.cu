/*
	

*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include "board.cu"


#define SPACE_LENGTH 5		// Spaces Size of rows / columns 
#define SPACE_WIDTH 5 
#define NUM_SPACES 25

#define WALL_LENGTH 4		// Walls size of rows/colums
#define WALL_WIDTH 4	
#define NUM_WALLS 16

#define POSSIBLE_DIRECTIONS 4 	// Possible directions for traversing/finding neighbors





/*	Parallel version of moveAllWalls. 
	2D thread array. 
	Input: walls, moves, opponentIdx

*/
__global__ void
solveForAllWalls(wall *d_walls, nextMove *d_moves, int oppPos) {
	int tidx = threadIdx.x;	// X-Dim = Wall
	int tidy = threadIdx.y;	// Y-Dim = Direction
	int space = blockIdx.x;	// Space #

	// Coalesced Load d_walls Global --> Shared for this Block
	// Only threads (0-15, 0)
	__shared__ wall sm_walls[NUM_WALLS];

	if (tidy == 0) {
		sm_walls[tidx] = d_walls[tidx];
	}

	// Create a blank board template --> Shared for this block
	// Only threads (0-15, 1)
	__shared__ space sm_boardTemplate[NUM_SPACES];

	// Spaces 0-15	First 16 spaces
	if (tidy == 1) {
		CUDA_boardInitParallel(&sm_boardTemplate, tidx);
	}
	// Spaces 16-29
	if (tidy = 2 && (tidx + 16) < NUM_SPACES) {
		CUDA_boardInitParallel(&sm_boardTemplate, (tidx + 16));
	}

	// Create shared move, global --> shared
	// Threads (0-4, 3)
	__shared__ nextMove move;

	if (tidy == 3) {

		switch(tidx) {
			case 0:
				move.space = d_moves[space].space;
				break;

			case 1:
				move.playerScore = d_moves[space].playerScore;
				break;

			case 2:
				move.oppScore = d_moves[space].oppScore;
				break;

			case 3:
				move.wallIdx = d_moves[space].wallIdx;
				break;

			case 4: 
				move.newDir = d_moves[space].newDir;
				break;
		}

	}

	__syncthreads();

	// Each thread makes local copy of walls
	wall l_walls[NUM_WALLS];
	memcpy(&l_walls, sm_walls, (sizeof(walls) * NUM_WALLS));


	// Check for wall collisions && if it's the same direction
	wall oldDir = l_walls[tidx];

	bool sameDir = (oldDir == (wall)tidy);
	bool collision = CUDA_checkWallCollisions(&l_walls, tidx);

	if (sameDir || collision) {
		// Do nothing and return
		return;
	}
	// If no collision, contune

	// Create a new board 
	space board[NUM_SPACES];

	CUDA_boardInitSeq(&board);

	// Initialize the board


	// Calculate shortest path for player

	// Calculate shortest path for opponent


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