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