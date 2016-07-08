/*	CPU Based Wallsolver 


*/

#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>


#define SPACE_LENGTH 5		// Spaces Size of rows / columns 
#define SPACE_WIDTH 5 

#define WALL_LENGTH 4		// Walls size of rows/colums
#define WALL_WIDTH 4	







typedef enum wall_dir {
	UP, DOWN, LEFT, RIGHT
} wall;


typedef struct space {
	bool up, down, left, right, start, finish;

} space;


void outputBoard(space *in) {

	for (int i = 0; i < SPACE_WIDTH; i++) {

		for (int j = 0; j < SPACE_LENGTH; j++) {
			int idx = (i * SPACE_WIDTH) + j;	// == board[i][j];

			printf("Space #: %d, UP: %d, DOWN: %d, LEFT: %d, RIGHT: %d \n", idx, in[idx].up, in[idx].down, in[idx].left, in[idx].right);

		}


	}

}


int main(int argc, char const *argv[])
{
	//

	space blankSpace = {false, false, false, false, false, false};

	int numSpaces = SPACE_LENGTH * SPACE_WIDTH;
	int spaceSize = sizeof(space) * numSpaces;

	int numWalls = WALL_LENGTH * WALL_WIDTH;
	int wallSize = sizeof(wall) * numWalls;

	// Malloc the array of wall / board
	wall *walls = malloc(wallSize);
	space *board = malloc(spaceSize);


	// Initialize, zero out the board 
	for (int i = 0; i < numSpaces; i++) {
		board[i] = blankSpace;
	}
	
	// Generate walls and board
	srand(1024);
	for (int i = 0; i < WALL_WIDTH; i++) {

		for (int j = 0; j < WALL_LENGTH; j++) {
			int idx = (i * WALL_LENGTH) + j; 	// == walls[i][j];

			//printf("wall idx: %d\n", idx);

			walls[i + j] = (wall)(rand() % 4);
			

			// Space idx that this wall affects
			// Per wall, figure out the adjacent spaces this wall effects
			// Set T/F value for each space depending how the wall effect its mobility through the maze
			int TL, TR, BL, BR;
			TL = idx + i;
			TR = TL + 1;
			BL = TL + SPACE_LENGTH;
			BR = BL + 1;


			board[TL].right = walls[idx] == UP;
			board[TL].down = walls[idx] == LEFT;

			board[TR].left = walls[idx] == UP;
			board[TR].down = walls[idx] == RIGHT;

			board[BL].right = walls[idx] == DOWN;
			board[BL].up = walls[idx] == LEFT;

			board[BR].left = walls[idx] == UP;
			board[BR].up = walls[idx] == RIGHT;


		}

	}

	board[0].start = true;
	board[numSpaces - 1].finish = true;

	// End Board Generation

	outputBoard(board);

	free(walls);
	free(board);








	return 0;
}



