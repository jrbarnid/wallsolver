/*
	Basic board functions and data structures 

*/

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <cuda_runtime.h>


#define SPACE_LENGTH 5		// Spaces Size of rows / columns 
#define SPACE_WIDTH 5 

#define WALL_LENGTH 4		// Walls size of rows/colums
#define WALL_WIDTH 4	


typedef enum wall {
	UP, DOWN, LEFT, RIGHT
} wall;

typedef struct space {
	bool up, down, left, right, start, finish;
} space;



void generateWalls(wall *walls) {
	/*	Randomly generate the walls for the board

	*/
	srand(1024);
	for (int i = 0; i < WALL_WIDTH; i++) {

		for (int j = 0; j < WALL_LENGTH; j++) {
			int idx = (i * WALL_LENGTH) + j; 	// == walls[i][j];

			walls[idx] = (wall)(rand() % 4);
			
			printf("IDX %d - %d\n", idx, walls[idx]);
		}

	}

	// Check for any wall collisions and re-randomize if necessary

	for (int i = 0; i < WALL_LENGTH; i++) {

		for (int j = 0; j < WALL_WIDTH; j++) {
			int idx = (i * WALL_WIDTH) + j;

			while (checkWallCollisions(walls, idx)) {
				printf("IDX No Overlap: %d - %d\n", idx, walls[idx]);
				walls[idx] = (wall)(rand() % 4);			
			}
		}
	}

}


void generateBoard(space *board, wall *walls) {
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
	idx = threadidx.x
*/
__device__ void
boardInit(space *board, int idx) {
	// mod = 0 == left edge space
	// mod = 4 == right edge space
	int i = idx / WALL_LENGTH;
	int j = idx % WALL_LENGTH;

	// Better to avoid divergence
	board[idx].up = (i != 0);
	board[idx].left = (j != 0);
	board[idx].down = (i != (SPACE_WIDTH - 1));
	board[idx].right = (j != (SPACE_LENGTH - 1));
	board[idx].start = false;
	board[idx].finish = false;

}

/*	Check if the current wall collides with neighboring walls
	Returns TRUE if there is a collision at wall IDX	
*/

__device__ bool
checkWallCollisions(wall *walls, int idx) {
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

/*


*/
__device__ void
moveWall(wall *walls, int idx, wall newDir) {

}

/*	Fix wall collisions on a randomly generated board
	Probably will not be used
*/
__device__ void
fixWallCollisions(wall *walls, int idx) {

	int i = idx / WALL_LENGTH;	// == row
	int j = idx % WALL_WIDTH;	// == col

	bool colUp = false;
	bool colDown = false;
	bool colLeft = false;
	bool colRight = false;
	bool starter = true;		// Used to start the while loop

	wall up, down, left, right;

	while (starter || colUp || colDown || colLeft || colRight) {
		printf("IDX No Overlap: %d - %d\n", idx, walls[idx]);

		starter = false;
		walls[idx] = (wall)(rand() % 4);

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


	}


}







