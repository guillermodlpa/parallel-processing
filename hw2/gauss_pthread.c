/* ------------------ HW2 --------------------- */

/* Guillermo de la Puente - A20314328                */
/* CS 546 - Parallel and Distributed Processing      */
/* Homework 2                                        */
/* Pthreads version of the Gaussian Elimination step */
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

/* Added includes for homework 2 */
#include <pthread.h>
#include <math.h>

/* Program Parameters */
#define MAXN 2000  /* Max value of N */
#define DIVFACTOR 32768.0
//#define DIVFACTOR 32768000.0
int N;  /* Matrix size */

/* Matrices and vectors */
volatile float A[MAXN][MAXN], B[MAXN], X[MAXN];
/* A * X = B, solve for X */

/* junk */
#define randm() 4|2[uid]&3

/* Prototype */
void gauss();  /* The function you will provide.
		* It is this routine that is timed.
		* It is called only on the parent.
		*/

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
  int row, col;

  printf("\nInitializing...\n");
  for (col = 0; col < N; col++) {
    for (row = 0; row < N; row++) {
      A[row][col] = (float)rand() / DIVFACTOR;
    }
    B[col] = (float)rand() / DIVFACTOR;
    X[col] = 0.0;
  }

}

/* Print input matrices */
void print_inputs() {
  int row, col;

  if (N < 10) {
    printf("\nA =\n\t");
    for (row = 0; row < N; row++) {
      for (col = 0; col < N; col++) {
	printf("%5.2f%s", A[row][col], (col < N-1) ? ", " : ";\n\t");
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

int main(int argc, char **argv) {
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
      /* Contrary to the man pages, this appears not to include the parent */
  printf("--------------------------------------------\n");
  
  exit(0);
}

/* ------------------ Above Was Provided --------------------- */

/****** You will replace this routine with your own parallel version *******/
/* Provided global variables are MAXN, N, A[][], B[], and X[],
 * defined in the beginning of this code.  X[] is initialized to zeros.
 *

/* ------------------ HW2 --------------------- */


/* Guillermo de la Puente - A20314328                */
/* CS 546 - Parallel and Distributed Processing      */
/* Homework 2                                        */
/* Pthreads version of the Gaussian Elimination step */
/*                                                   */
/* 2014 Spring semester                              */
/* Professor Zhiling Lan                             */
/* TA Eduardo Berrocal                               */
/* Illinois Institute of Technology                  */


/* Select the number of worker threads:              */
int numWorkers = 4;


/* Barrier initialization and method */
pthread_mutex_t barrier_mutex;
pthread_cond_t go;
int numArrived =0;

/* Barrier method */
/* It's used to synchronize all threads after normalizing one column */
void barrier(){

  pthread_mutex_lock(&barrier_mutex);
  numArrived++;

  if (numArrived<numWorkers) {
    pthread_cond_wait(&go, &barrier_mutex);
  }
  else {
    pthread_cond_broadcast(&go);
    numArrived=0; /* be prepared for next barrier */
  }
  pthread_mutex_unlock(&barrier_mutex);
}

/* Method that performs the gaussian elimination parallel process */
void * gaussianElimination(void *s) {

  int norm, row, col;  /* Normalization row, and zeroing
      * element row and col */
  int from, to, numRows, threadid, position;
  float multiplier;
//int multiplier;

  threadid = *((int*)(&s));;

  for (norm = 0; norm < N - 1; norm++) {

    /* Chunk represents the number of elements that this current thread will process */
    numRows = (N - norm - 1);

    /* Because 'from' and 'to' are int, the "+0.5f" is used to approximate to the closest integer, instead of truncate */
    /* The *10, +5 and /10 are used to approximate to the closest integer without recurring to float operations        */
    /* For example: 
        float case --->  6.0/8.0 = 0.75  
        int case   --->  6  /8   = 0
 
        our case --->    (((6*10) / 8)+5 )/10 =  (60/8 + 5) / 10 = (7 + 5) / 10 = 1  = round(0.75)
    */
    from = ( 10*norm + ( ( threadid     * numRows) *10 / numWorkers ) + 5)/10;
    to   = ( 10*norm + ( ( (threadid+1) * numRows) *10 / numWorkers ) + 5)/10;

    /* Similar code than in the sequential case */
    for (row = from + 1; row <= to; row++) {

      multiplier = A[row][norm] / A[norm][norm];

      //multiplier = A[row][norm]*100 / A[norm][norm];

      for (col = norm; col < N; col++) {
        A[row][col] -= A[norm][col] * multiplier;
        //A[row][col] -= A[norm][col] * multiplier / 100;
      }

      B[row] -= B[norm] * multiplier;
    }

    /* Threads barrier. All of them need to reach this point before going to the next iteration */
    barrier();
  }
}


void gauss() {

  printf("Computing using Pthreads.\n");

  /* Create threads */
  pthread_t threads[numWorkers];

  /* Set up barrier and condition */
  pthread_mutex_init(&barrier_mutex, NULL);
  pthread_cond_init(&go, NULL);

  /* Initialize threads */
  int i;
  for ( i = 0; i < numWorkers; i++ )
    pthread_create(&threads[i], NULL, gaussianElimination, (void*)i);

  /* Wait for all threads to finish */
  for ( i = 0; i < numWorkers; i++ )
    pthread_join(threads[i], NULL);


  int norm, row, col;  /* Normalization row, and zeroing
      * element row and col */

  /* Back substitution */
  for (row = N - 1; row >= 0; row--) {
    X[row] = B[row];

    for (col = N-1; col > row; col--) {
      X[row] -= A[row][col] * X[col];
    }
    
    X[row] /= A[row][row];
  }
}

/*
int main(int argc, char **argv) {

  int rank, size;
  MPI_Init (&argc, &argv); 
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &size); 
  mainMethod(argc,argv);
  MPI_Finalize();
  return 0;
}*/