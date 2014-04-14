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

float *test;



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
	/*A = (float*)malloc( N*N*sizeof(float) );
	B = (float*)malloc( N*sizeof(float) );
	X = (float*)malloc( N*sizeof(float) );*/
	test = (float*) malloc( N*sizeof(float));
}

/* Free allocated memory for arrays */
void free_memory() {
	/*free(A);
	free(B);
	free(X);*/
	free(test);
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

    /*printf("\nProcess number %d of %d says hi\n",
            my_rank+1, p);*/

    gauss();

    free_memory();

	MPI_Finalize();
}

/* Main function that performs the algorithms */
void gauss() {

	MPI_Status status;

    if ( my_rank == SOURCE ) {
		int i,k;
		for (k = 0; k < N; k++) test[k] = 1;
    	for ( i = 1; i < p; i++ ) {
    		MPI_Send( &test[0], N, MPI_FLOAT, i,0, MPI_COMM_WORLD );
    	}
    }
    else
		MPI_Recv( &test[0], N, MPI_FLOAT, SOURCE, 0, MPI_COMM_WORLD, &status);


	int k;
	for (k = 0; k < N; k++) test[k] += 0.01f;


	printf("\nProcess number %d of %d says: got %5.2f, %5.2f, %5.2f, %5.2f\n",
        my_rank+1, p, test[0], test[1], test[2], test[3]);


	if ( my_rank != SOURCE )
		MPI_Send( &test[0], N, MPI_FLOAT, SOURCE,0, MPI_COMM_WORLD );
	else {
		int i;
		float *local_test = (float*) malloc( N*sizeof(float));
		for ( i = 1; i < p; i++ ) {
    		MPI_Recv( &local_test[0], N, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &status);
    		int k;
			for (k = 0; k < N; k++) test[k] += local_test[k];
    	}
    	printf("\nProcess number %d of %d says: got %5.2f, %5.2f, %5.2f, %5.2f\n",
        	my_rank+1, p, test[0], test[1], test[2], test[3]);
    }


}
