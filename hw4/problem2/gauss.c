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
	printf("%5.2f%s", A[row + N*col], (col < N-1) ? ", " : ";\n\t");
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
	printf("%5.2f%s", A[row + N*col], (col < N-1) ? ", " : ";\n\t");
      }
    }
  }
}

/* Initialize A and B (and X to 0.0s) */
void initialize_inputs() {

  int row, col;

  printf("\nInitializing...\n");
  for (col = 0; col < N; col++) {
    for (row = 0; row < N; row++) {
      A[row + N*col] = (float)rand() / DIVFACTOR;
    }
    B[col] = (float)rand() / DIVFACTOR;
    X[col] = 0.0;
  }

}


int main(int argc, char **argv) {

	/* Prototype function*/
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
		print_inputs();
	}

	/* Free memory used for the arrays that we allocated previously */
    free_memory();

    /* The barrier prevents any process to reach the finalize before the others have finished their communications */
    MPI_Barrier(MPI_COMM_WORLD);

	MPI_Finalize();
}

/* Main function that performs the algorithms */
void gauss() {

	MPI_Status status;
	int row, col, i, norm;

	for (norm = 0; norm < N - 1; norm++) {

		float A_norm = A[norm + N*norm];

		/* Broadcast the A[norm][norm] */
		MPI_Bcast( &A_norm, 1, MPI_FLOAT, SOURCE, MPI_COMM_WORLD );

		printf("\nProcess number %d of %d says: my A[norm][norm] is %5.2f\n",
        	my_rank+1, p, A_norm);
	}

}



void gauss2() {

	MPI_Status status;
	int row, col, i;

	/* Sender side */
    if ( my_rank == SOURCE ) {
		for (col = 0; col < N; col++) {
			B[col] = 0;
			for (row = 0; row < N; row++)
				A[row + N*col] = 1;
		}

    	for ( i = 1; i < p; i++ ) {
    		MPI_Send( &A[0], N*N, MPI_FLOAT, i,0, MPI_COMM_WORLD );
    		MPI_Send( &B[0], N, MPI_FLOAT, i,0, MPI_COMM_WORLD );
    	}

    }
    /* Receiver side */
    else {
		MPI_Recv( &A[0], N*N, MPI_FLOAT, SOURCE, 0, MPI_COMM_WORLD, &status);
		MPI_Recv( &B[0], N, MPI_FLOAT, SOURCE, 0, MPI_COMM_WORLD, &status);
	}


	for (col = 0; col < N; col++) {
		B[col] += 0.1f;
		for (row = 0; row < N; row++)
			A[row + N*col] += 0.01f;
	}


	/*printf("\nProcess number %d of %d says: got %5.2f, %5.2f, %5.2f, %5.2f\n",
        my_rank+1, p, A[0], A[1], A[2], A[3]);*/

	/* Sender side */
	if ( my_rank != SOURCE ) {
		MPI_Send( &A[0], N*N, MPI_FLOAT, SOURCE,0, MPI_COMM_WORLD );
		MPI_Send( &B[0], N, MPI_FLOAT, SOURCE,0, MPI_COMM_WORLD );
	}
	/* Receiver side */
	else {
		int i;
		float *local_A = (float*) malloc( N*N*sizeof(float));
		float *local_B = (float*) malloc( N*sizeof(float));
		for ( i = 1; i < p; i++ ) {
    		MPI_Recv( &local_A[0], N*N, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &status);
    		MPI_Recv( &local_B[0], N, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &status);

			for (col = 0; col < N; col++) {
				B[col] += local_B[col];
				for (row = 0; row < N; row++)
					A[row + N*col] += local_A[row + N*col];
			}
    	}
    	free(local_A);
    	free(local_B);
    	/*printf("\nProcess number %d of %d says: got %5.2f, %5.2f, %5.2f, %5.2f\n",
        	my_rank+1, p, A[0], A[1], A[2], A[3]);*/
    }


}
