/*
	Basic board functions and data structures 

*/

#include <stdio.h>
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


__device__ void
generateBoard(space *board, wall *walls, int idx) {

	

}

/*	Parallel coalesced board initialization
	Blanks out the board
	idx = threadidx.x
*/
__device__ void
initBoard(space *board, int idx) {
	// mod = 0 == left edge space
	// mod = 4 == right edge space
	int mod = idx % SPACE_LENGTH;

	// Better to avoid divergence
	board[idx].up = !(i < SPACE_LENGTH);
	board[idx].left = (mod != 0);

	board[idx].down = !(i > 19);
	board[idx].right = (mod != 4);

	board[idx].start = (idx == 0);
	board[idx].finish = (idx == 29);

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







