/* ------------------ HW3 --------------------- */

/* Guillermo de la Puente - A20314328                */
/* CS 546 - Parallel and Distributed Processing      */
/* Homework 4                                        */
/* Part 2/2                                          */
/* Gaussian elimination without pivoting with MPI    */
/*                                                   */
/* 2014 Spring semester                              */
/* Professor Zhiling Lan                             */
/* TA Eduardo Berrocal                               */
/* Illinois Institute of Technology                  */


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
#include <time.h>

#include <mpi.h>

#define MAXN 4000 /* Max value of N */
int N; /* Matrix size */
#define DIVFACTOR 327680000.0f

#define SOURCE 0

/* My process rank           */
int my_rank;
/* The number of processes   */
int p; 

/* Matrixes given by a pointer */
float *A, *B, *X;



/* returns a seed for srand based on the time */
unsigned int time_seed() {
  struct timeval t;
  struct timezone tzdummy;

  gettimeofday(&t, &tzdummy);
  return (unsigned int)(t.tv_usec);
}


/* Set the program parameters from the command-line arguments */
void parameters(int argc, char **argv) {
  int seed = 0;  /* Random seed */
  char uid[32]; /*User name */

  /* Read command-line arguments */
  srand(time_seed());  /* Randomize */

  if (argc == 3) {
    seed = atoi(argv[2]);
    srand(seed);
    if ( my_rank == SOURCE ) printf("Random seed = %i\n", seed);
  } 
  if (argc >= 2) {
    N = atoi(argv[1]);
    if (N < 1 || N > MAXN) {
      if ( my_rank == SOURCE ) printf("N = %i is out of range.\n", N);
      exit(0);
    }
  }
  else {
    if ( my_rank == SOURCE ) printf("Usage: %s <matrix_dimension> [random seed]\n",
           argv[0]);    
    exit(0);
  }

  /* Print parameters */
  if ( my_rank == SOURCE )  printf("\nMatrix dimension N = %i.\n", N);
}


/* Allocates memory for A, B and X */
void allocate_memory() {
	A = (float*)malloc( N*N*sizeof(float) );
	B = (float*)malloc( N*sizeof(float) );
	X = (float*)malloc( N*sizeof(float) );
}

/* Free allocated memory for arrays */
void free_memory() {
	free(A);
	free(B);
	free(X);
}


/* Print input matrices */
void print_inputs() {
  int row, col;

  if (N < 10) {
    printf("\nA =\n\t");
    for (row = 0; row < N; row++) {
      for (col = 0; col < N; col++) {
	printf("%5.2f%s", A[col+N*row], (col < N-1) ? ", " : ";\n\t");
      }
    }
    printf("\nB = [");
    for (col = 0; col < N; col++) {
      printf("%5.2f%s", B[col], (col < N-1) ? "; " : "]\n");
    }
  }
}

/* Print matrix A */
void print_A() {
  int row, col;

  if (N < 10) {
    printf("\nA =\n\t");
    for (row = 0; row < N; row++) {
      for (col = 0; col < N; col++) {
	printf("%5.2f%s", A[col+N*row], (col < N-1) ? ", " : ";\n\t");
      }
    }
  }
}
/* Print matrix B */
void print_B() {
  int col;
  if (N < 10) {
	  printf("\nB = [");
	    for (col = 0; col < N; col++) {
	      printf("%5.2f%s", B[col], (col < N-1) ? "; " : "]\n");
	    }
	}
}
/* Print matrix X */
void print_X() {
  int row;

  if (N < 100) {
    printf("\nX = [");
    for (row = 0; row < N; row++) {
      printf("%5.2f%s", X[row], (row < N-1) ? "; " : "]\n");
    }
  }
}

/* Initialize A and B (and X to 0.0s) */
void initialize_inputs() {

  int row, col;

  printf("\nInitializing...\n");
  for (row = 0; row < N; row++) {
    for (col = 0; col < N; col++) {
      A[col+N*row] = (float)rand() / DIVFACTOR;
    }
    B[row] = (float)rand() / DIVFACTOR;
    X[row] = 0.0;
  }

}


int main(int argc, char **argv) {

	/* Prototype functions*/
	void gauss();

	MPI_Init(&argc, &argv);

	/* Get my process rank */
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	/* Find out how many processes are being used */
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    /* Every process reads the parameters to prepare dimension */
    parameters(argc, argv);

    /* Every process must allocate memory for the arrays */
    allocate_memory();

    if ( my_rank == SOURCE ) {
	    /* Initialize A and B */
		initialize_inputs();

		/* Print input matrices */
		print_inputs();
	}

    /*printf("\nProcess number %d of %d says hi\n",
            my_rank+1, p);*/

	gauss();

    if ( my_rank == SOURCE ) {

		/* Print input matrices */
		print_A();
		print_B();
		print_X();
	}


    /* The barrier prevents any process to reach the finalize before the others have finished their communications */
    MPI_Barrier(MPI_COMM_WORLD);

	/* Free memory used for the arrays that we allocated previously */
    //free_memory();

	MPI_Finalize();
}


/* Includes both algorithms */
void gauss() {

	void gaussElimination();
	void backSubstitution();

	/* Times */
	double t1, t2;

	/* Barrier to sync all processes before starting the algorithms */
	MPI_Barrier(MPI_COMM_WORLD);

	/* Initial time */
	if ( my_rank == SOURCE )
		t1 = MPI_Wtime();

	/* Gauss Elimination is performed using MPI */
	gaussElimination();

	/* Back Substitution is performed sequentially */
	if ( my_rank == SOURCE ) {
		backSubstitution();

		/* Finish time */
		t2 = MPI_Wtime();

		printf("\nElapsed time: %f miliseconds\n", (t2-t1) * 1000 );
	}
	
}



/* Guassian Elimination algorithm using MPI */
void gaussElimination() {

	MPI_Status status;
	int row, col, i, norm;
	float multiplier;

	/* Main loop. After every iteration, a new column will have all 0 values down the [norm] index */
	for (norm = 0; norm < N - 1; norm++) {

		/* --------------------------------------- */
		/* 	Broadcasting of common values          */
    	/* 	-------------------------------------- */
		/* Broadcast the A[norm] row and B[norm], important values of this iteration */
		MPI_Bcast( &A[ N*norm ], N, MPI_FLOAT, SOURCE, MPI_COMM_WORLD );
		MPI_Bcast( &B[norm], 1, MPI_FLOAT, SOURCE, MPI_COMM_WORLD );



		/* ---------------------------------------   */
		/* 	Calculation of number of rows to operate */
    	/* 	--------------------------------------   */
		/* subset of rows of this iteration */
    	int subset = N - 1 - norm;
    	/* number that indicates the step as a float */
    	float step = ((float)subset ) / p;
    	/* First and last rows that this process will work into for this iteration */
    	int local_row_a = norm + 1 + ceil( step * my_rank );
    	int local_row_b = norm + 1 + floor( step * (my_rank+1) );
    	int number_of_rows = local_row_b - local_row_a +1;




    	/* --------------------------------------- */
		/* 	Send data from process 0 to others     */
    	/* 	-------------------------------------- */
    	if ( my_rank == SOURCE ) {

    		int remote_row_a = 0; 
    		int remote_row_b = 0;
    		int number_of_rows_r = 0;
    		for ( i = 1; i < p; i++ ) {

    			int previous_remote_row_a = remote_row_a;
    			int previous_remote_row_b = remote_row_b;

    			/* We send to each process the amount of data that they are going to handle */
    			remote_row_a = norm + 1 + ceil( step * i );
		    	remote_row_b = norm + 1 + floor( step * (i+1) );
		    	number_of_rows_r = remote_row_b - remote_row_a +1;

		    	/*printf("\nPrpces %d of %d says in iteration %d that STAGE1 REMOTE a=%d, b=%d and n=%d\n",
					        my_rank+1, p, norm+1,remote_row_a,remote_row_b,number_of_rows_r) ;*/

		    	/* In case this process isn't assigned any task, continue. This happens when there are more processors than rows */
		    	if ( number_of_rows_r > 0  && remote_row_a < N) {

		    		MPI_Send( &A[remote_row_a * N], N * number_of_rows_r, MPI_FLOAT, i,0, MPI_COMM_WORLD );
		    		MPI_Send( &B[remote_row_a],         number_of_rows_r, MPI_FLOAT, i,0, MPI_COMM_WORLD );
		    	}
	    	}
    	}
    	/* Receiver side */
    	else {

    		if ( number_of_rows > 0  && local_row_a < N) {

	    		MPI_Recv( &A[local_row_a * N], N * number_of_rows, MPI_FLOAT, SOURCE, 0, MPI_COMM_WORLD, &status);
	    		MPI_Recv( &B[local_row_a],         number_of_rows, MPI_FLOAT, SOURCE, 0, MPI_COMM_WORLD, &status);
	    	}
    	}


    	
	    /*printf("\nProcess %d: Iteration number %d of %d\n",
			        my_rank, norm+1, N-1);
		print_A();*/



		/* --------------------------------------- */
		/* 	Gaussian elimination                   */
		/* 	The arrays only have the needed values */
    	/* 	-------------------------------------- */
		if ( number_of_rows > 0  && local_row_a < N) {	
			/* Similar code than in the sequential case */
			for (row = local_row_a; row <= local_row_b; row++) {

		 		multiplier = A[N*row + norm] / A[norm + N*norm];
				for (col = norm; col < N; col++) {
		 			A[col+N*row] -= A[N*norm + col] * multiplier;
		 		}

		 		B[row] -= B[norm] * multiplier;
			}
		}
    	


    	/* --------------------------------------- */
		/* 	Send back the results                  */
    	/* 	-------------------------------------- */

    	/* Sender side */
    	if ( my_rank != SOURCE ) {



    		if ( number_of_rows > 0  && local_row_a < N) {

				printf("\nProcess %d iteration %d OUT a=%d, b=%d and n=%d\n",
					my_rank, norm,local_row_a,local_row_b,number_of_rows) ;

    			MPI_Send( &A[local_row_a * N], N * number_of_rows, MPI_FLOAT, SOURCE,0, MPI_COMM_WORLD );
    			//MPI_Send( &A[local_row_a * N], N * number_of_rows, MPI_FLOAT, SOURCE,0, MPI_COMM_WORLD );
	    		//MPI_Send( &B[local_row_a],         number_of_rows, MPI_FLOAT, SOURCE,0, MPI_COMM_WORLD );
    		}
    	}
    	/* Receiver side */
    	else {

    		int remote_row_a = 0; 
    		int remote_row_b = 0;
    		int number_of_rows_r = 0;
    		for ( i = 1; i < p; i++ ) {

    			int previous_remote_row_a = remote_row_a;
    			int previous_remote_row_b = remote_row_b;

    			/* We send to each process the amount of data that they are going to handle */
    			remote_row_a = norm + 1 + ceil( step * i );
		    	remote_row_b = norm + 1 + floor( step * (i+1) );
		    	number_of_rows_r = remote_row_b - remote_row_a +1;

		    	/* In case this process isn't assigned any task, continue. This happens when there are more processors than rows */
		    	if ( number_of_rows_r > 0  && remote_row_a < N) {

			    	printf("\nProcess %d iteration %d IN  a=%d, b=%d and n=%d\n",
						        my_rank, norm,remote_row_a,remote_row_b,number_of_rows_r) ;

			    	MPI_Recv( &A[remote_row_a * N], N * number_of_rows_r, MPI_FLOAT, i,0, MPI_COMM_WORLD, &status );
		    		//MPI_Recv( &A[remote_row_a * N], N * number_of_rows_r, MPI_FLOAT, i,0, MPI_COMM_WORLD, &status );
		    		//MPI_Recv( &B[remote_row_a],         number_of_rows_r, MPI_FLOAT, i,0, MPI_COMM_WORLD, &status );
			    }
	    	}

	    	/* Trace to see the progress of the algorithm iteration after iteration */
			/*printf("\nIteration number %d of %d\n",
			        norm+1, N-1);
	    	print_A();*/
    	}

    }
}


/* Back substitution sequential algorithm */
void backSubstitution () {

	int norm, row, col;  /* Normalization row, and zeroing
      * element row and col */

	/* Back substitution */
	for (row = N - 1; row >= 0; row--) {
		X[row] = B[row];

		for (col = N-1; col > row; col--) {
			X[row] -= A[N*row+col] * X[col];
		}
	    
		X[row] /= A[N*row + row];
	}
}
