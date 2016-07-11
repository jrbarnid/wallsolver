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
generateBoard(space *board, wall *walls, int tid) {

	

}

