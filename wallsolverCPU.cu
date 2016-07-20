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


/* These are for an old way of tracking time */
struct timezone Idunno;	
struct timeval startTime, endTime;


/* 
	set a checkpoint and show the (natural) running time in seconds 
*/
double report_running_time() {
	long sec_diff, usec_diff;
	gettimeofday(&endTime, &Idunno);
	sec_diff = endTime.tv_sec - startTime.tv_sec;
	usec_diff= endTime.tv_usec-startTime.tv_usec;
	if(usec_diff < 0) {
		sec_diff --;
		usec_diff += 1000000;
	}
	printf("Running time for CPU version: %ld.%06ld\n", sec_diff, usec_diff);
	return (double)(sec_diff*1.0 + usec_diff/1000000.0);
}



bool checkWallCollisions(wall *walls, int idx) {
	/* 	Make sure no walls overlap
		For each wall, identify the neighboring walls if they exist
		determine if a neighbor caused a conflict
		Return TRUE if there is a collision
	*/
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


void generateBoard(space *board, wall *walls) {
	/* 	Generate the board
		For each wall, identify the board spaces that it effects
		Determine the effect of each affected space's mobility
	*/
	int numSpaces = SPACE_LENGTH * SPACE_WIDTH;

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


void boardInit(space *board) {
	// Initialize the board, blank
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
			board[idx].parent = -1;
			board[idx].distance = 0;
			board[idx].state = UNEXPLORED;
		}

	}
}


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


void outputBoard(space *in) {

	for (int i = 0; i < SPACE_WIDTH; i++) {

		for (int j = 0; j < SPACE_LENGTH; j++) {
			int idx = (i * SPACE_WIDTH) + j;	// == board[i][j];

			printf("Space #: %d, UP: %d, DOWN: %d, LEFT: %d, RIGHT: %d \n", idx, in[idx].up, in[idx].down, in[idx].left, in[idx].right);

		}

	}

}

/* Used strictly for debugging. This should not be used in the final version */
void printAdjList(int adjList[][POSSIBLE_DIRECTIONS]) {
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
void initializeAdjList(int adjList[][POSSIBLE_DIRECTIONS]) {
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

nextSpace findMinimum(space *in, int adjList[][POSSIBLE_DIRECTIONS], int idx) {
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

int shortestPath(space *in, int idxIn = 0) {
	int adjList[SPACE_LENGTH*SPACE_WIDTH][POSSIBLE_DIRECTIONS];
	initializeAdjList(adjList);
	int i = idxIn;
	nextSpace next;
	
	// Iterate through the board until we reach the finish node.
	while (!in[i].finish) {
		int j;
		int minimum = 99999;
		
		// Run greedy shortest path on all of the current space's neighbors.
		/*
		in[i].state = VISITED;
		int tmp = i;
		for (j = 0; j < POSSIBLE_DIRECTIONS; ++j) {
			if (adjList[i][j] == -1)
				continue;

			next = findMinimum(in, adjList, i);
			if (next.distance < minimum) {
				i = next.index;
			} 
		}
		*/

		in[i].state = VISITED;
		int tmp = i;
		next = findMinimum(in, adjList, i);
		i = next.index;

		if (i == -1) {
			i = in[tmp].parent;
		}
		else {
			in[i].parent = tmp;
			in[i].distance = in[in[i].parent].distance + next.distance;
		}
	}

	printf("Total distance: %d\n", in[i].distance);
	while (!in[i].start) {
		printf("Space #%d\n", i);
		i = in[i].parent;
	}
	printf("Space #%d\n", i);

	return in[i].distance;
}

/*

*/
int moveWall(wall *in, int idx, wall newDir, int pos) {
	// Local copy of the walls
	// Set the walls[idx] to the new direction

	wall oldDir = in[idx];

	// return -1 if the new dir = old dir
	if (in[idx] == newDir) return -1;	

	in[idx] = newDir;
	// Return -1 if wall move results in collision
	if ( checkWallCollisions(in, idx) ) in[idx] = oldDir; return -1;	

	space *board = (space *) malloc( sizeof(space) * (SPACE_LENGTH * SPACE_WIDTH) );

	printf("-- moveWall: wallidx: %d --\n", idx);

	boardInit(board);
	generateBoard(board, in);
	shortestPath(board, pos);

	free(board);

	// Reset to the old direction
	in[idx] = oldDir;
	
	return 0;
}

void moveAllWalls(space *board, wall *walls, int playerIdx, int oppIdx, nextMove *results) {
	int score = 0;
	
	nextMove bestForPlayer = {shortestPath(board, playerIdx), playerIdx, -1, -1};
	nextMove worstForOpponent = {shortestPath(board, oppIdx), oppIdx, -1, -1};

	for (int i = 0; i < (WALL_LENGTH * WALL_WIDTH); i++) {

		for (int j = 0; j < 3; j++) {
			// Measure player value
			score = moveWall(walls, i, (wall) j, playerIdx);
			if (score < bestForPlayer.score) {
				bestForPlayer.score = score;
				bestForPlayer.nextSpace = playerIdx;
				bestForPlayer.wallIdx = i;
				bestForPlayer.newDir = (wall) j;
			}

			// Measure opponent value
			score = moveWall(walls, i, (wall) j, oppIdx);
			if (score > worstForOpponent.score) {
				worstForOpponent.score = score;
				worstForOpponent.nextSpace = oppIdx;
				worstForOpponent.wallIdx = i;
				worstForOpponent.newDir = (wall) j;
			}
		}

	}

	results[0] = bestForPlayer;
	results[1] = worstForOpponent;
}





int main(int argc, char const *argv[])
{
	
	int numSpaces = SPACE_LENGTH * SPACE_WIDTH;
	int spaceSize = sizeof(space) * numSpaces;

	int numWalls = WALL_LENGTH * WALL_WIDTH;
	int wallSize = sizeof(wall) * numWalls;

	// Malloc the array of wall / board
	wall *walls = (wall *)malloc(wallSize);
	space *board = (space *)malloc(spaceSize);

	// Initialize, zero out the board 
	boardInit(board);

	// Generate walls 
	generateWalls(walls);

	generateBoard(board, walls);

	// Start the timer
	gettimeofday(&startTime, &Idunno);

	outputBoard(board);
	shortestPath(board);


	// results 0 = player 	1 = opponent
	nextMove *results = (nextMove *) malloc(sizeof(nextMove) * 2);
	// board, walls, playerIdx to be moved to, current opponent idx, results
	moveAllWalls(board, walls, 0, 0, results);

	// Report the running time
	report_running_time();

	free(walls);
	free(board);








	return 0;
}



