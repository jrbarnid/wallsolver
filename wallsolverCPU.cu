/*	CPU Based Wallsolver 

	nvcc wallsolverCPU.cu -o testCPU

*/

#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>






#define SPACE_LENGTH 5		// Spaces Size of rows / columns 
#define SPACE_WIDTH 5 
#define NUM_SPACES 25

#define WALL_LENGTH 4		// Walls size of rows/colums
#define WALL_WIDTH 4	
#define NUM_WALLS 16

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
	int space;
	int playerScore;
	int oppScore;
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

			//printf("Maze Generated: %d - %d\n", idx, walls[idx]);

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
	printf("Initialize Board\n");
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

			printf("outputBoard: Space #: %d, UP: %d, DOWN: %d, LEFT: %d, RIGHT: %d \n", idx, in[idx].up, in[idx].down, in[idx].left, in[idx].right);

		}

	}

}


void outputResults(nextMove *results, int numResults) {
	/*	Output the best valued move for each space
	
	*/
	printf("----- RESULTS -----\n");
	for (int i = 0; i < numResults; i++) {
		printf("Best Move for Space %d\n", results[i].space);
		printf("Move wall %d to direction %d\n", results[i].wallIdx, results[i].newDir);
		printf("Player Score: %d, Opponent Score: %d\n\n", results[i].playerScore, results[i].oppScore);
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


void resetSpaces(space *in) {
	int i;
	int numSpaces = SPACE_LENGTH * SPACE_WIDTH;

	for (i = 0; i < numSpaces; ++i) {
		in[i].parent = -1;
		in[i].state = UNEXPLORED;
	}
	
	return;
}

int shortestPath(space *in, int idxIn = 0) {

	int adjList[SPACE_LENGTH*SPACE_WIDTH][POSSIBLE_DIRECTIONS];
	initializeAdjList(adjList);
	int i = idxIn;
	nextSpace next;
	int distance;

	// If shortestPath is used multiple times then we need to reset the parent & state.
	resetSpaces(in);
	
	// Iterate through the board until we reach the finish node.
	while (!in[i].finish) {
		// Run greedy shortest path on all of the current space's neighbors.
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

	distance = in[i].distance;
	printf("Total distance: %d\n", distance);
	while (!in[i].start) {
		printf("Space #%d\n", i);
		i = in[i].parent;
	}
	printf("Space #%d\n", i);

	return distance;
}



/*

*/

void moveWall(wall *in, int wallIdx, wall newDir, nextMove *move, int moveIdx, int oppPos) {

	wall oldDir = in[wallIdx];			// Temp store the old wall direction

	bool sameDir = in[wallIdx] == newDir;		// Set the walls[idx] to the new direction

	in[wallIdx] = newDir;
	bool collision = checkWallCollisions(in, wallIdx);

	// If same direction or a collision
	// Reset to old direction and return -1
	if (sameDir || collision) {
		in[wallIdx] = oldDir;
		return;
	}
	
	printf("No collision - Space: %d, WallID: %d\n", move[moveIdx].space, wallIdx);

	space *board = (space *)malloc(sizeof(space) * NUM_SPACES);

	boardInit(board);
	generateBoard(board, in);

	int playerScore = 0;
	playerScore = shortestPath(board, move[moveIdx].space);

	int oppScore = 0;
	oppScore = shortestPath(board, oppPos);


	printf("Space: %d, WallID: %d, Wall New Dir: %d, Player Score: %d, Opp Score: %d\n", move[moveIdx].space, wallIdx, newDir, playerScore, oppScore);


	if (playerScore < move[moveIdx].playerScore || oppScore > move[moveIdx].oppScore) {
		move[moveIdx].playerScore = playerScore;
		move[moveIdx].oppScore = oppScore;
		move[moveIdx].wallIdx = wallIdx;
		move[moveIdx].newDir = newDir;
	}

	// Reset to the old direction
	in[wallIdx] = oldDir;

	free(board);
	
}


void moveAllWalls(wall *walls, int oppIdx, nextMove *results, int resultsIdx) {
	/*	For all walls, orient them each possible direction
		Check for collisions or if same direction
		If not, see if it changes the player or opponent's shortest path

	*/
	for (int i = 0; i < (WALL_LENGTH * WALL_WIDTH); i++) {

		for (int j = 0; j < 3; j++) {
			
			// Check player (walls, wallID, direction, playerSpace, oppSpace)
			moveWall(walls, i, (wall) j, results, resultsIdx, oppIdx);

		}

	}

}





/* 
***********************************************************
*** Inputs: space *in (our board)                         *
***         idx (the space # we're finding neighbors for) *
***********************************************************
*** Output: 12-element int array with indexes to all      *
***         potential neighbors. -1 is entered for any    *
***         elements that are unnecessary.                *
***********************************************************
*** Purpose: Find all possible neighbors a space can      *
***          visit on the current turn. If a wall is      *
***          encountered after a step is taken then the   *
***          adjacent neighbors are added to the array.   *
***********************************************************
*/ 
int* findNeighbors(space *in, int idx) {
	int const NUM_NEIGHBORS = 12;
	int *neighbors = (int *)malloc(NUM_NEIGHBORS*sizeof(int));
	int const MAX_DISTANCE = 3;
	int numSpaces = SPACE_LENGTH * SPACE_WIDTH;
	int count = 0;
	int i = idx;
	int neighborIdx = 0;
	int const WALL_COST = 3;

	// Grab neighbors going up
	while (count < MAX_DISTANCE) {
		// Don't gather upward neighbors if index is in the top row.
		if (idx < SPACE_WIDTH)
			break;
		
		i -= SPACE_WIDTH;
		if (i >= numSpaces || i < 0) {
			break;
		}
		else if (in[i].down && count > 0) {
			// Get left neighbors if we're reached a wall
			i += SPACE_WIDTH;
			int tmp = i;
			int tmpCount = count;
			while (count < MAX_DISTANCE) {
				i--;

				if (in[i].right || (i % SPACE_WIDTH) == 0) {
					count += MAX_DISTANCE;
					break;
				}
				else {
					neighbors[neighborIdx++] = i;
					count++;
				}
			}

			// Get right neighbors if we've reached a wall
			i = tmp;
			count = tmpCount;
			while (count < MAX_DISTANCE) {
				i++;
				
				if (in[i].left || (i % SPACE_WIDTH == (SPACE_WIDTH - 1))) {
					count += MAX_DISTANCE;
					break;
				}
				else {
					neighbors[neighborIdx++] = i;
					count++;
				}
			}
		}
		else if (in[i].down && count == 0) {
			count += WALL_COST;
			neighbors[neighborIdx++] = i;
		}
		else {
			count++;
			neighbors[neighborIdx++] = i;
		}
	}

	// Grab neighbors going down
	count = 0;
	i = idx;
	while (count < MAX_DISTANCE) {
		// Don't grab downward neighbors if index is in the bottom row.
		if (i >= (numSpaces - SPACE_WIDTH))
			break;

		i += SPACE_WIDTH;
		if (i >= numSpaces || i < 0) {
			break;
		}
		else if (in[i].up && count > 0) {
			// Get left neighbors if we're reached a wall
			i -= SPACE_WIDTH;
			int tmp = i;
			int tmpCount = count;
			while (count < MAX_DISTANCE) {
				i--;

				if (in[i].right || (i % SPACE_WIDTH) == 0) {
					count += MAX_DISTANCE;
					break;
				}
				else {
					neighbors[neighborIdx++] = i;
					count++;
				}
			}

			// Get right neighbors if we've reached a wall
			i = tmp;
			count = tmpCount;
			while (count < MAX_DISTANCE) {
				i++;
				
				if (in[i].left || (i % SPACE_WIDTH == (SPACE_WIDTH - 1))) {
					count += MAX_DISTANCE;
					break;
				}
				else {
					neighbors[neighborIdx++] = i;
					count++;
				}
			}
		}
		else if (in[i].up && count == 0) {
			count += WALL_COST;
			neighbors[neighborIdx++] = i;
		}
		else {
			count++;
			neighbors[neighborIdx++] = i;
		}
 
	}

	// Grab neighbors going left
	count = 0;
	i = idx;
	while (count < MAX_DISTANCE) {
		// Don't gather leftward neighbors if index is in the left column.
		if ((idx % SPACE_WIDTH == 0))
			break;
		
		i--;
		if (i >= numSpaces || i < 0) {
			break;
		}
		else if (in[i].right && count > 0) {
			// Get up neighbors if we're reached a wall
			i++;
			int tmp = i;
			int tmpCount = count;
			while (count < MAX_DISTANCE) {
				i -= SPACE_WIDTH;

				if (i < 0 || in[i].down) {
					count += MAX_DISTANCE;
					break;
				}
				else {
					neighbors[neighborIdx++] = i;
					count++;
				}
			}

			// Get down neighbors if we've reached a wall
			i = tmp;
			count = tmpCount;
			while (count < MAX_DISTANCE) {
				i += SPACE_WIDTH;
				
				if (in[i].up || (i >= (numSpaces - SPACE_WIDTH))) {
					count += MAX_DISTANCE;
					break;
				}
				else {
					neighbors[neighborIdx++] = i;
					count++;
				}
			}
		}
		else if (in[i].right && count == 0) {
			count += WALL_COST;
			neighbors[neighborIdx++] = i;
		}
		else {
			count++;
			neighbors[neighborIdx++] = i;
		}
	}

	// Grab neighbors going right
	count = 0;
	i = idx;
	while (count < MAX_DISTANCE) {
		// Don't gather rightward neighbors if index is in the right column.
		if (idx % SPACE_WIDTH == (SPACE_WIDTH - 1))
			break;
		
		i++;
		if (i >= numSpaces || i < 0) {
			break;
		}
		else if (in[i].left && count > 0) {
			// Get up neighbors if we're reached a wall
			i--;
			int tmp = i;
			int tmpCount = count;
			while (count < MAX_DISTANCE) {
				i -= SPACE_WIDTH;

				if (i < 0 || in[i].down) {
					count += MAX_DISTANCE;
					break;
				}
				else {
					neighbors[neighborIdx++] = i;
					count++;
				}
			}

			// Get down neighbors if we've reached a wall
			i = tmp;
			count = tmpCount;
			while (count < MAX_DISTANCE) {
				i += SPACE_WIDTH;
				
				if (in[i].up || (i >= (numSpaces - SPACE_WIDTH))) {
					count += MAX_DISTANCE;
					break;
				}
				else {
					neighbors[neighborIdx++] = i;
					count++;
				}
			}
		}
		else if (in[i].left && count == 0) {
			count += WALL_COST;
			neighbors[neighborIdx++] = i;
		}
		else {
			count++;
			neighbors[neighborIdx++] = i;
		}
	}

	// Set all unused spaces in our return array to -1 to indicate they don't exist
	for (i = neighborIdx; i < NUM_NEIGHBORS; ++i)
		neighbors[i] = -1;
	
	return neighbors;
}


nextMove *getPossibleMoves(space *board, int pos, int *numSpaces) {
	// Counter number of possible spaces @ pos
	int *neighbors = findNeighbors(board, pos);


	int possibleSpaces = 0;
	for(int i = 0; i < 12; i++) {
		printf("Neighbors for space %d: %d\n", pos, neighbors[i]);

		if (neighbors[i] != -1) possibleSpaces++;
	}

	nextMove *moves = (nextMove *) malloc( sizeof(nextMove) * possibleSpaces );

	int j = 0;
	for(int i = 0; i < 12 && j < possibleSpaces; i++) {
		if (neighbors[i] == -1 ) break;

		moves[j].space = neighbors[i];
		moves[j].playerScore = 100;		// Intentionally high preset
		moves[j].oppScore = -1;
		moves[j].wallIdx = -1;
		moves[j].newDir = (wall) 0;

		j++;

	}

	free(neighbors);

	//numSpaces = possibleSpaces;
	return moves;
}


nextMove pickBestMove(nextMove *moves, int possibleSpaces) {
	nextMove best;
	
	if (possibleSpaces > 0) {
		best.space = moves[0].space;
		best.playerScore = moves[0].playerScore;
		best.oppScore = moves[0].oppScore;
		best.wallIdx = moves[0].wallIdx;
		best.newDir = moves[0].newDir;
	}

	int diff = best.oppScore - best.playerScore;
	int i;
	for (i = 0; i < possibleSpaces; ++i) {
		int tmpdiff = moves[i].oppScore - moves[i].playerScore;
		if (tmpdiff > diff) {
			best.space = moves[i].space;
			best.playerScore = moves[i].playerScore;
			best.oppScore = moves[i].oppScore;
			best.wallIdx = moves[i].wallIdx;
			best.newDir = moves[i].newDir;
			diff = tmpdiff;
		}
	}

	return best;
}



int main(int argc, char const *argv[])
{

	int playerPos = 0;
	int oppPos = 0;

	
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

	// Testing
	//shortestPath(board, 0);
	//shortestPath(board, 0);


	// board, walls, playerIdx to be moved to, current opponent idx, results
	//moveAllWalls(board, walls, 0, 0, results);

	// Report the running time
	//report_running_time();


	// Get neighbors of a space
	
	int *neighbors = findNeighbors(board, 7);
	int i;
	for (i = 0; i < 12; ++i) {
		printf("Neighbors for space #7: %d\n", neighbors[i]);
	}
	neighbors = findNeighbors(board, 17);
	for (i = 0; i < 12; ++i) {
		printf("Neighbors for space #17: %d\n", neighbors[i]);
	}

	neighbors = findNeighbors(board, playerPos);
	for (i = 0; i < 12; ++i) {
		printf("Neighbors for space #%d: %d\n", playerPos, neighbors[i]);
	}


	
	// Count the number of possible spaces = # of blocks
	int possibleSpaces = 0;
	for (int i = 0; i < 12; i++) {
		if (neighbors[i] != -1) {
			possibleSpaces++;
		}
	}

	// Malloc an array nextMove[ # of neighbors ]
	nextMove *moves = (nextMove *)malloc( sizeof(nextMove) * possibleSpaces );
	printf("DEBUG: successful malloc of results\n");


	// Zero-out the results array and set each move.space ot the neighbor space
	int j = 0;
	for (int i = 0; i < 12 && j < possibleSpaces; i++) {
		if (neighbors[i] != -1) {
			printf("Init results array. Moves[%d], Space: %d\n", j, neighbors[i]);

			moves[j].space = neighbors[i];
			moves[j].playerScore = 100;		// Intentionally high preset
			moves[j].oppScore = -1;
			moves[j].wallIdx = -1;
			moves[j].newDir = (wall) 0;

			j++;
		}
	}

	

	// TODO: DEBUG this function
	//int *possibleSpaces = getPossibleMoves(board, playerPos, possibleSpaces);



	/* start counting time */
	gettimeofday(&startTime, &Idunno);


	// Set the nextSpace in the array to each part in array
	/*	For each possible space --> Move all 16 walls all possible direction
		Determine the shortest path
	*/
	for (int i = 0; i < possibleSpaces; i++) {
		moveAllWalls(walls, oppPos, moves, i);
	}


	/* check the total running time */ 
	report_running_time();


	
	
	outputResults(moves, possibleSpaces);
	nextMove bestMove = pickBestMove(moves, possibleSpaces);
	printf("Best Move: %d\n", bestMove.space);


	printf("Memory Sizes - walls: %d, board: %d, nextMove (maximum): %d, neighbors: %d\n", wallSize, spaceSize, sizeof(nextMove)*12, sizeof(neighbors));

	free(walls);
	free(board);
	//free(possibleSpaces);
	free(neighbors);
	free(moves);


	return 0;
}



