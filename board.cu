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

#define POSSIBLE_DIRECTIONS 4 	// Possible directions for traversing/finding neighbors

typedef enum wall {
	UP, DOWN, LEFT, RIGHT
} wall;

typedef enum status {
	UNEXPLORED, VISITED, EXPLORED
} status;

typedef struct space {
	bool up, down, left, right, start, finish;
	int parent;
	int distance;
	status state;
} space;

typedef struct nextSpace {
	int index;
	int distance;
} nextSpace;

typedef struct nextMove {
	int score;
	int nextSpace;
	int wallIdx;
	wall newDir;
} nextMove;



__global__ void
CUDA_calculateShortestPath() {


}

/*	Check if the current wall collides with neighboring walls
	Returns TRUE if there is a collision at wall IDX	
*/
<<<<<<< HEAD
__device__ bool checkWallCollisions(wall *walls, int idx) {
=======


__device__ bool 
CUDA_checkWallCollisions(wall *walls, int idx) {
>>>>>>> master
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
CUDA_generateBoardCUDA(space *board, wall *walls) {
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
CUDA_boardInitParallel(space *board, int idx) {
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
<<<<<<< HEAD

=======
__device__ void
CUDA_boardInitSeq(space *board, int idx) {
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
/*
	tid = wall to be moved and tested

*/
__device__ void
CUDA_moveWallParallel(wall *in, int idx, int playerIdx, int oppIdx, nextMove *results) {

	for (int j = 0; j < 3; j++) {
		// Local copy of the walls
		// Set the walls[idx] to the new direction
		wall oldDir = in[idx];

		bool sameDir = in[idx] == (wall) j;

		in[idx] = newDir;
		bool collision = checkWallCollisions(in, idx);

		// If NOT sameDir OR Collision
		if (!(sameDir || collision)) {

			space board[(SPACE_LENGTH * SPACE_WIDTH)];

			boardInitSeq(&board[0]);
			generateBoardCUDA(&board[0], in);

			int playerScore = shortestPath(board, playerIdx);
			int oppScore = shortestPath(board, oppIdx);
			
			if (score < results[0].score) {
				results[0].score = score;
				results[0].nextSpace = playerIdx;
				results[0].wallIdx = i;
				results[0].newDir = (wall) j;
			}

			if (score > results[1].score) {
				results[1].score = score;
				results[1].nextSpace = oppIdx;
				results[1].wallIdx = i;
				results[1].newDir = (wall) j;
			}
			
		}

		// Reset to the old direction
		in[idx] = oldDir;

	}

}
>>>>>>> master


__device__ void 
CUDA_moveWallParallel2D(wall *in, int wallIdx, int newDir, int playerPos, int oppPos) {

	wall oldDir = in[wallIdx];
	bool sameDir = in[wallIdx] == oldDir;

	in[wallIdx] = newDir;
	bool collision = CUDA_checkWallCollisions(in, wallIdx);

	/* 	If not the same direction or nor collisions
		
	*/
	if (!(sameDir || collision)) {
		space board[SPACE_LENGTH * SPACE_WIDTH];

		CUDA_boardInitSeq(&board[0]);
		CUDA_generateBoardCUDA(&board[0]);

		int playerScore = shortestPath(&board[0], playerPos);
		int oppScore = shortestPath(&board[0], oppPos);

	}

	in[wallIdx] = oldDir;
}






