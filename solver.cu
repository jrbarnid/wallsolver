/*
	

*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include "board.cu"


/*	Parallel version of moveAllWalls. 
	2D thread array. 
	Input: walls, moves, opponentIdx

*/
__global__ void
solveForAllWalls(wall *d_walls, nextMove *d_moves, int oppPos) {
	int tidx = threadIdx.x;	// X-Dim = Wall
	int tidy = threadIdx.y;	// Y-Dim = Direction
	int idx = blockIdx.x;	// Space #

	// Coalesced Load d_walls Global --> Shared for this Block
	// Only threads (0-15, 0)
	__shared__ wall sharedWalls[NUM_WALLS];

	if (tidy == 0) {
		sharedWalls[tidx] = d_walls[tidx];
	}

	// Create a blank board template --> Shared for this block
	// Only threads (0-15, 1)
	extern __shared__ space sharedBoardTemplate[];

	// Spaces 0-15	First 16 spaces
	if (tidy == 1) {
		CUDA_boardInitParallel(sharedBoardTemplate, tidx);
	}
	// Spaces 16-29
	if (tidy = 2 && (tidx + 16) < NUM_SPACES) {
		CUDA_boardInitParallel(sharedBoardTemplate, (tidx + NUM_WALLS));
	}

	// Create shared move, global --> shared
	// Threads (0-4, 3)
	__shared__ nextMove move;

	if (tidy == 3) {

		switch(tidx) {
			case 0:
				move.space = d_moves[idx].space;
				break;

			case 1:
				move.playerScore = d_moves[idx].playerScore;
				break;

			case 2:
				move.oppScore = d_moves[idx].oppScore;
				break;

			case 3:
				move.wallIdx = d_moves[idx].wallIdx;
				break;

			case 4: 
				move.newDir = d_moves[idx].newDir;
				break;
		}

	}

	__syncthreads();

	// Each thread makes local copy of walls
	wall l_walls[NUM_WALLS];
	memcpy(&l_walls, sharedWalls, (sizeof(wall) * NUM_WALLS));


	// Check for wall collisions && if it's the same direction
	wall oldDir = l_walls[tidx];

	bool sameDir = (oldDir == (wall)tidy);
	bool collision = CUDA_checkWallCollisions(&l_walls[0], tidx);

	if (sameDir || collision) {
		// Do nothing and return
		return;
	}
	// If no collision, contune

	// Create a new board 
	space board[NUM_SPACES];

	CUDA_boardInitSeq(board);

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
	int playerPos = 0;
	int oppPos = 0;
	
	int numSpaces = SPACE_LENGTH * SPACE_WIDTH;
	size_t spaceSize = sizeof(space) * numSpaces;

	int numWalls = WALL_LENGTH * WALL_WIDTH;
	size_t wallSize = sizeof(wall) * numWalls;


	// Malloc the array of wall / board
	wall *walls = (wall *)malloc(wallSize);
	space *board = (space *)malloc(spaceSize);



	// Initialize and setup the current board state
	boardInit(board);
	generateWalls(walls);
	generateBoard(board, walls);


	// Find nearest neighbors to player


	// Determine the number of spaces around the player


	// Malloc memory to store next possible moves



	// Malloc space on device, copy to device
	wall *d_walls = NULL;

	checkCudaError( cudaMalloc((void**) &d_walls, wallSize), 
		"Malloc d_walls");

	// cudaMemcpy(target, source, size, function)
	checkCudaError( cudaMemcpy(d_walls, walls, wallSize, cudaMemcpyHostToDevice), 
		"Copy walls to device");



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
