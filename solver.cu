/*
	

*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include "board.h"

/*	Parallel version of moveAllWalls. 
	2D thread array. 
	Input: walls, moves, opponentIdx

*/
__global__ void
CUDA_solveForAllWalls(wall *d_walls, nextMove *d_moves, int oppPos) {
	int tidx = threadIdx.x;		// X-Dim = Wall
	int tidy = threadIdx.y;		// Y-Dim = Direction
	int idx = blockIdx.x;		// Space #

	printf("Thread: (%d, %d)\n", tidx, tidy);
	// - - - - - 
	// Coalesced Load d_walls Global --> Shared for this Block
	// Only threads (0-15, 0)
	__shared__ wall sharedWalls[NUM_WALLS];

	if (tidy == 0) {
		sharedWalls[tidx] = d_walls[tidx];
	}


	// - - - - -
	// Create a blank board template --> Shared for this block
	// Only threads (0-15, 1)
	__shared__ space sharedBoardTemplate[NUM_SPACES];

	// Spaces 0-15	First 16 spaces
	if (tidy == 1) {
		CUDA_boardInitParallel(sharedBoardTemplate, tidx);
	}
	// Spaces 16-29
	if (tidy = 2 && (tidx + 16) < NUM_SPACES) {
		printf("tidx: %d; NUM_SPACES: %d\n", (tidx + 16), NUM_SPACES);
		CUDA_boardInitParallel(sharedBoardTemplate, (tidx + 16));
	}


	// - - - - - 
	// Create shared move, global --> shared
	// Threads (0-4, 3)
	__shared__ nextMove move;

	if (tidy == 3) {

		switch(tidx) {		// Thread [] 
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


	// Create local copy of new board 
	space l_board[NUM_SPACES];
	memcpy(&l_board, sharedBoardTemplate, sizeof(space) * NUM_SPACES);

	// Generate the board from the walls
	CUDA_generateBoard(&l_board[0], l_walls);


	// Calculate shortest path for player & opponent
	int playerScore = CUDA_shortestPath(&l_board[0], move.space);
	int oppScore = CUDA_shortestPath(&l_board[0], oppPos);

	printf("PlayerPos: %d -- PlayerScore: %d -- OppScore: %d\n", move.space, playerScore, oppScore);
	if (playerScore < move.playerScore || oppScore > move.oppScore) {
		move.playerScore = playerScore;
		move.oppScore = oppScore;
		move.wallIdx = tidx;
		move.newDir = (wall) tidy;
	}

} 


// CUDA Error Check
void checkCudaError(cudaError_t e, char const *in) {
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
	int *neighbors = findNeighbors(board, playerPos);

	// Determine the number of spaces around the player
	// Count the number of possible spaces = # of blocks
	int possibleSpaces = 0;
	for (int i = 0; i < 12; i++) {
		if (neighbors[i] != -1) {
			possibleSpaces++;
		}
	}

	// Malloc an array nextMove[ # of neighbors ]
	nextMove *moves = (nextMove *)malloc( sizeof(nextMove) * possibleSpaces );

	// Zero-out the results array and set each move.space ot the neighbor space
	int j = 0;
	for (int i = 0; i < 12 && j < possibleSpaces; i++) {
		if (neighbors[i] != -1) {
			printf("Init results array. Moves[%d], Space: %d\n", j, neighbors[i]);

			moves[j].space = neighbors[i];
			moves[j].playerScore = 100;		// Intentionally high preset
			moves[j].oppScore = -1;
			moves[j].wallIdx = -1;
			moves[j].newDir = (wall) 0;

			j++;
		}
	}




	// Malloc space on device, copy to device
	wall *d_walls = NULL;
	nextMove *d_moves = NULL;

	checkCudaError( cudaMalloc((void**) &d_walls, wallSize), 
		"Malloc d_walls");
	checkCudaError( cudaMalloc((void**) &d_moves, (sizeof(nextMove) * possibleSpaces) ), 
		"Malloc d_walls");

	// cudaMemcpy(target, source, size, function)
	checkCudaError( cudaMemcpy(d_walls, walls, wallSize, cudaMemcpyHostToDevice), 
		"Copy walls to device");
	checkCudaError( cudaMemcpy(d_moves, moves, (sizeof(nextMove) * possibleSpaces), cudaMemcpyHostToDevice), 
		"Copy moves to device");




	// Setup: Measure Runtime
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	/*	Kernel Call
		Blocks = possible spaces
		Threads = #walls * #possible directions

	*/
	dim3 grid(16,4);
	CUDA_solveForAllWalls <<<possibleSpaces, grid>>> (d_walls, d_moves, oppPos);

	checkCudaError(cudaGetLastError(), "Checking Last Error, Kernel Launch");

	// Report kernel runtime
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Time to generate: %0.5f ms\n", elapsedTime );

	cudaEventDestroy(start);
	cudaEventDestroy(stop);


	// Copy Device --> Host
	// cudaMemcpy(target, source, size, function)
	checkCudaError( cudaMemcpy(moves, d_moves, (sizeof(nextMove) * possibleSpaces), cudaMemcpyDeviceToHost), 
		"Copy moves to host");


	outputResults(moves, possibleSpaces);


	// Free Memory
	checkCudaError(cudaFree(d_walls), "Free device histogram");
	checkCudaError(cudaFree(d_moves), "Free device atom_list");

	free(board);
	free(walls);
	free(moves);

	return 0;
}
