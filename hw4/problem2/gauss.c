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
    printf("Random seed = %i\n", seed);
  } 
  if (argc >= 2) {
    N = atoi(argv[1]);
    if (N < 1 || N > MAXN) {
      printf("N = %i is out of range.\n", N);
      exit(0);
    }
  }
  else {
    printf("Usage: %s <matrix_dimension> [random seed]\n",
           argv[0]);    
    exit(0);
  }

  /* Print parameters */
  printf("\nMatrix dimension N = %i.\n", N);
}





/* Initialize A and B (and X to 0.0s) */
void initialize_inputs() {

	A = (float*)malloc( N*N*sizeof(float) );
	B = (float*)malloc( N*sizeof(float) );
	X = (float*)malloc( N*sizeof(float) );

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
void print_X() {
  int row;

  if (N < 100) {
    printf("\nX = [");
    for (row = 0; row < N; row++) {
      printf("%5.2f%s", X[row], (row < N-1) ? "; " : "]\n");
    }
  }
}


/* Used for MPI_Init() */
int argcG;
char **argvG;

/* Gaussian elimination algorithm */
/* The parallelization is made in the second loop */
/* This is probably not optimal because it requires a lot of communication */
void gaussianElimination() {

	int my_rank;   /* My process rank           */
    int p;         /* The number of processes   */


	/* Get my process rank */
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	/* Find out how many processes are being used */
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    printf("Process number %d of %d says hi\n",
            my_rank+1, p);

	
}


/* Main function that performs the algorithms */
void gauss() {

	printf("Computing in parallel using MPI.\n");

	int norm, row, col;  /* Normalization row, and zeroing
			* element row and col */
	float multiplier;

	/* Gaussian elimination */
	/*for (norm = 0; norm < N - 1; norm++) {
		for (row = norm + 1; row < N; row++) {
			multiplier = A[row][norm] / A[norm][norm];
			for (col = norm; col < N; col++) {
				A[row][col] -= A[norm][col] * multiplier;
			}
			B[row] -= B[norm] * multiplier;
		}
	}*/
	gaussianElimination();

	/* (Diagonal elements are not normalized to 1.  This is treated in back
	* substitution.)
	*/

	/* Back substitution */
	for (row = N - 1; row >= 0; row--) {
		X[row] = B[row];
		for (col = N-1; col > row; col--) {
			X[row] -= A[row + col*N] * X[col];
		}
		X[row] /= A[row + N*row];
	}
}



int main(int argc, char **argv) {


	MPI_Init(&argc, &argv);

	/* Timing variables */
	struct timeval etstart, etstop;  /* Elapsed times using gettimeofday() */
	struct timezone tzdummy;
	clock_t etstart2, etstop2;  /* Elapsed times using times() */
	unsigned long long usecstart, usecstop;
	struct tms cputstart, cputstop;  /* CPU times for my processes */

 	/* Process program parameters */
	parameters(argc, argv);

	/* Initialize A and B */
	initialize_inputs();

	/* Print input matrices */
	print_inputs();

	/* Start Clock */
	printf("\nStarting clock.\n");
	gettimeofday(&etstart, &tzdummy);
	etstart2 = times(&cputstart);

	/* Gaussian Elimination */
	gauss();

	/* Stop Clock */
	gettimeofday(&etstop, &tzdummy);
	etstop2 = times(&cputstop);
	printf("Stopped clock.\n");
	usecstart = (unsigned long long)etstart.tv_sec * 1000000 + etstart.tv_usec;
	usecstop = (unsigned long long)etstop.tv_sec * 1000000 + etstop.tv_usec;

	/* Comment to see result of elmination */
	printf("After gaussian elimination");
	print_inputs();

	/* Display output */
	print_X();

    /* Free allocated memory */
	free_memory();


	/* Display timing results */
	printf("\nElapsed time = %g ms.\n",
		(float)(usecstop - usecstart)/(float)1000);

	printf("(CPU times are accurate to the nearest %g ms)\n",
		1.0/(float)CLOCKS_PER_SEC * 1000.0);
	printf("My total CPU time for parent = %g ms.\n",
		(float)( (cputstop.tms_utime + cputstop.tms_stime) -
			(cputstart.tms_utime + cputstart.tms_stime) ) /
			(float)CLOCKS_PER_SEC * 1000);
	printf("My system CPU time for parent = %g ms.\n",
		(float)(cputstop.tms_stime - cputstart.tms_stime) /
		(float)CLOCKS_PER_SEC * 1000);
	printf("My total CPU time for child processes = %g ms.\n",
		(float)( (cputstop.tms_cutime + cputstop.tms_cstime) -
			(cputstart.tms_cutime + cputstart.tms_cstime) ) /
			(float)CLOCKS_PER_SEC * 1000);
	printf("--------------------------------------------\n");

	MPI_Finalize();
  
	exit(0);
}


