/*	CPU Based Wallsolver 

	nvcc wallsolverCPU.cu -o testCPU

*/

#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>


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

	//space blankSpace = {false, false, false, false, false, false};
	space blankSpace = {true, true, true, true, false, false};

	int numSpaces = SPACE_LENGTH * SPACE_WIDTH;
	int spaceSize = sizeof(space) * numSpaces;

	int numWalls = WALL_LENGTH * WALL_WIDTH;
	int wallSize = sizeof(wall) * numWalls;

	// Malloc the array of wall / board
	wall *walls = (wall *)malloc(wallSize);
	space *board = (space *)malloc(spaceSize);


	// Initialize, zero out the board 
	/*
	for (int i = 0; i < numSpaces; i++) {
		board[i] = blankSpace;
	}
	*/
	for (int i = 0; i < SPACE_LENGTH; i++) {

		for (int j = 0; j < SPACE_WIDTH; j++) {
			int idx = (i * SPACE_WIDTH) + j;
			//board[idx] = blankSpace;

			/*
			if (i == 0) board[idx].up = false;
			if (j == 0) board[idx].left = false;
			if (i == (SPACE_WIDTH - 1)) board[idx].down = false;
			if (j == (SPACE_LENGTH - 1)) board[idx].right = false;
			*/

			// Better to avoid divergence
			board[idx].up = (i != 0);
			board[idx].left = (j != 0);
			board[idx].down = (i != (SPACE_WIDTH - 1));
			board[idx].right = (j != (SPACE_LENGTH - 1));
			board[idx].start = false;
			board[idx].finish = false;
		}

	}


	
	// Generate walls and board
	srand(1024);
	for (int i = 0; i < WALL_WIDTH; i++) {

		for (int j = 0; j < WALL_LENGTH; j++) {
			int idx = (i * WALL_LENGTH) + j; 	// == walls[i][j];

			walls[idx] = (wall)(rand() % 4);
			
			printf("IDX %d - %d\n", idx, walls[idx]);
		}

	}

	/* 	Make sure no walls overlap
		For each wall, identify the neighboring walls if they exist
		determine if a neighbor caused a conflict and while conflicts
		exist, randomize a new wall
	*/
	for (int i = 0; i < WALL_WIDTH; i++) {

		for (int j = 0; j < WALL_LENGTH; j++) {
			int idx = (i * WALL_LENGTH) + j;

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
				down = walls[idx + WALL_LENGTH];
				colDown = (walls[idx] == DOWN) && (down == UP);
			} 

			if (i > 0) {
				up = walls[idx - WALL_LENGTH];
				colUp = (walls[idx] == UP) && (up == DOWN);
			}


			while (colUp || colDown || colLeft || colRight) {
				printf("IDX No Overlap: %d - %d\n", idx, walls[idx]);
				walls[idx] = (wall)(rand() % 4);

				if (j < 4) {
					colRight = (walls[idx] == RIGHT) && (right == LEFT);
				}

				if (j > 0) {
					colLeft = (walls[idx] == LEFT) && (left == RIGHT);
				}

				if (i < 4) {
					colDown = (walls[idx] == DOWN) && (down == UP);
				} 

				if (i > 0) {
					colUp = (walls[idx] == UP) && (up == DOWN);
				}


			}

		}

	}



	generateBoard(board, walls);


	outputBoard(board);

	free(walls);
	free(board);








	return 0;
}



