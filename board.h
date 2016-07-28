#ifndef BOARD_CU
#define BOARD_CU

/*
	Basic board functions and data structures 

*/

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <cuda_runtime.h>
#include "boardCPU.h"



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

/* Used strictly for debugging. This should not be used in the final version */
__device__ void 
CUDA_printAdjList(int adjList[][POSSIBLE_DIRECTIONS]) {
	int i = 0;
	int numSpaces = SPACE_LENGTH * SPACE_WIDTH;

	for (i = 0; i < numSpaces; ++i) {
		printf("Space #%d's neighbors: UP: %d, DOWN: %d, LEFT: %d, RIGHT: %d \n", i, 
				adjList[i][0], adjList[i][1], adjList[i][2], adjList[i][3]);
	}
}

/* Set all neighbors to -1, then cycle through and add neighbors for each space
   All spaces marked with -1 afterwards means the neighbor is invalid and can be ignored
*/
__device__ void 
CUDA_initializeAdjList(int adjList[][POSSIBLE_DIRECTIONS]) {
	int i = 0;
	int numSpaces = SPACE_LENGTH * SPACE_WIDTH;

	for (i = 0; i < numSpaces; ++i) {
		int j;
		for (j = 0; j < POSSIBLE_DIRECTIONS; ++j) {
			adjList[i][j] = -1;
		}
	}
	
	for (i = 0; i < numSpaces; ++i) {
		// Add up neighbor to list
		if (i >= SPACE_WIDTH)
			adjList[i][0] = i - SPACE_LENGTH;
		
		// Add down neighbor to list
		if (i < (numSpaces - SPACE_WIDTH))
			adjList[i][1] = i + SPACE_LENGTH;
		
		// Add left neighbor to list
		if (i % SPACE_WIDTH != 0)
			adjList[i][2] = i - 1;

		// Add right neighbor to list 
		if (i % SPACE_WIDTH != (SPACE_WIDTH - 1))
			adjList[i][3] = i + 1;
	}
	// printAdjList(adjList);
}

__device__ nextSpace 
CUDA_findMinimum(space *in, int adjList[][POSSIBLE_DIRECTIONS], int idx) {
	int min = 9999;
	int min_idx = -1;
	int j;
	const int WALL_COST = 3;
	nextSpace next;
	
	// Find the best next step based on our index's neighbors.
	for (j = 0; j < POSSIBLE_DIRECTIONS; ++j) {
		if (adjList[idx][j] == -1 || in[adjList[idx][j]].state == VISITED)
			continue;

		if (j == 0) {
			if (in[idx].up && min > WALL_COST) {
				min = WALL_COST;
				min_idx = adjList[idx][j];
			}
			else if (!in[idx].up && min > 1) {
				min = 1;
				min_idx = adjList[idx][j];
			}
		}
		if (j == 1) { 
			if (in[idx].down && WALL_COST <= min) { 
				min = WALL_COST;
				min_idx = adjList[idx][j];
			}
			else if (!in[idx].down && min > 1) {
				min = 1;
				min_idx = adjList[idx][j];
			}
		}
		if (j == 2) {
			if (in[idx].left && min > WALL_COST) {
				min = WALL_COST;
				min_idx = adjList[idx][j];
			}
			else if (!in[idx].left && min > 1) {
				min = 1;
				min_idx = adjList[idx][j];
			}
		}
		if (j == 3) {
			if (in[idx].right && min > WALL_COST) {
				min = WALL_COST;
				min_idx = adjList[idx][j];
			}
			else if (!in[idx].right && min > 1) {
				min = 1;
				min_idx = adjList[idx][j];
			}
		}
	}

	next.index = min_idx;
	next.distance = min;

	return next;
}


__device__ void 
CUDA_resetSpaces(space *in) {
	int i;
	int numSpaces = SPACE_LENGTH * SPACE_WIDTH;

	for (i = 0; i < numSpaces; ++i) {
		in[i].parent = -1;
		in[i].state = UNEXPLORED;
	}
	
	return;
}

__device__ int 
CUDA_shortestPath(space *in, int idxIn = 0) {

	int adjList[SPACE_LENGTH*SPACE_WIDTH][POSSIBLE_DIRECTIONS];
	CUDA_initializeAdjList(adjList);
	int i = idxIn;
	nextSpace next;
	int distance;

	// If shortestPath is used multiple times then we need to reset the parent & state.
	CUDA_resetSpaces(in);
	
	// Iterate through the board until we reach the finish node.
	while (!in[i].finish) {
		// Run greedy shortest path on all of the current space's neighbors.
		in[i].state = VISITED;
		int tmp = i;
		next = CUDA_findMinimum(in, adjList, i);
		i = next.index;

		if (i == -1) {
			i = in[tmp].parent;
		}
		else {
			in[i].parent = tmp;
			in[i].distance = in[in[i].parent].distance + next.distance;
		}
	}

	distance = in[i].distance;
	printf("Total distance: %d\n", distance);
	while (!in[i].start) {
		printf("Space #%d\n", i);
		i = in[i].parent;
	}
	printf("Space #%d\n", i);

	return distance;
}




#endif // BOARD_CU
