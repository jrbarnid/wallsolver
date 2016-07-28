#ifndef BOARD_CU
#define BOARD_CU

/*
	Basic board functions and data structures 

*/

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <cuda_runtime.h>
#include "boardCPU.cu"



__device__ int
CUDA_shortestPath(space *board, int pos) {

	return 0;
}

/*	Check if the current wall collides with neighboring walls
	Returns TRUE if there is a collision at wall IDX	
*/

__device__ bool 
CUDA_checkWallCollisions(wall *walls, int idx) {

	int i = idx / WALL_LENGTH;
	int j = idx % WALL_WIDTH;

	bool colUp = false;
	bool colDown = false;
	bool colLeft = false;
	bool colRight = false;

	wall up, down, left, right;

	if (j < 4) {
		right = walls[idx + 1];
		colRight = (walls[idx] == RIGHT) && (right == LEFT);
	}

	if (j > 0) {
		left = walls[idx - 1];
		colLeft = (walls[idx] == LEFT) && (left == RIGHT);
	}

	if (i < 4) {
		down = walls[idx + WALL_WIDTH];
		colDown = (walls[idx] == DOWN) && (down == UP);
	} 

	if (i > 0) {
		up = walls[idx - WALL_LENGTH];
		colUp = (walls[idx] == UP) && (up == DOWN);
	}

	// Returns true if there is a collision
	return (colUp || colDown || colLeft || colRight);
}





__device__ void 
CUDA_generateBoard(space *board, wall *walls) {
	/* 	Generate the board
		For each wall, identify the board spaces that it effects
		Determine the effect of each affected space's mobility
	*/
	int numSpaces = WALL_LENGTH * WALL_WIDTH;

	for (int i = 0; i < WALL_WIDTH; i++) {

		for (int j = 0; j < WALL_LENGTH; j++) {
			int idx = (i * WALL_LENGTH) + j;

			printf("Maze Generated: %d - %d\n", idx, walls[idx]);

			// Determine the 4 adjacent spaces to this wall
			int TL = idx + i;
			int TR = TL +1;
			int BL = TL + SPACE_LENGTH;
			int BR = BL +1;

			if (board[TL].right) board[TL].right = (walls[idx] != UP);
			if (board[TL].down) board[TL].down = (walls[idx] != LEFT);

			if (board[TR].left) board[TR].left = board[TL].right;
			if (board[TR].down) board[TR].down = (walls[idx] != RIGHT);

			if (board[BL].right) board[BL].right = (walls[idx] != DOWN);
			if (board[BL].up) board[BL].up = board[TL].down;

			if (board[BR].left) board[BR].left = board[BL].right;
			if (board[BR].up) board[BR].up = board[TR].down;

		}

	}

	board[0].start = true;
	board[numSpaces - 1].finish = true;

}



/*	Parallel coalesced board initialization
	Blanks out the board
	idx = space to solve for
*/
__device__ void
CUDA_boardInitParallel(space *board, int idx) {

	if (idx >= NUM_SPACES) return;

	// mod = 0 == left edge space
	// mod = 4 == right edge space
	int i = idx / WALL_LENGTH;
	int j = idx % WALL_LENGTH;

	// Better to avoid divergence
	board[idx].up = (i != 0);
	board[idx].left = (j != 0);
	board[idx].down = (i != (SPACE_WIDTH - 1));
	board[idx].right = (j != (SPACE_LENGTH - 1));
	board[idx].finish = false;
	board[idx].parent = -1;
	board[idx].distance = 0;
	board[idx].state = UNEXPLORED;
}



__device__ void
CUDA_boardInitSeq(space *board) {
	// mod = 0 == left edge space
	// mod = 4 == right edge space
	int i, j;
	for (int idx = 0; idx < NUM_SPACES; i++) {
		i = idx / WALL_LENGTH;
		j = idx % WALL_LENGTH;

		// Better to avoid divergence
		board[idx].up = (i != 0);
		board[idx].left = (j != 0);
		board[idx].down = (i != (SPACE_WIDTH - 1));
		board[idx].right = (j != (SPACE_LENGTH - 1));
		board[idx].finish = false;
		board[idx].parent = -1;
		board[idx].distance = 0;
		board[idx].state = UNEXPLORED;
	}

}






#endif // BOARD_CU