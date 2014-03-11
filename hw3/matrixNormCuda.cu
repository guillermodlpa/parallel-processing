/* Matrix normalization.
 * Compile with "gcc matrixNorm.c" 
 */

/* ****** ADD YOUR CODE AT THE END OF THIS FILE. ******
 * You need not submit the provided code.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
#include <time.h>
#include <iostream>

/* Program Parameters */
#define MAXN 8000  /* Max value of N */
#define DIVISOR 3276800000.0
//#define DIVISOR 327680000.0
int N;  /* Matrix size */

/* Matrices */
float A[MAXN][MAXN], B[MAXN][MAXN];

/* junk */
#define randm() 4|2[uid]&3

/* Prototype */
void matrixNorm();

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

/* Initialize A and B*/
void initialize_inputs() {
  int row, col;

  printf("\nInitializing...\n");
  for (col = 0; col < N; col++) {
    for (row = 0; row < N; row++) {
      A[row][col] = (float)rand() / DIVISOR;
      B[row][col] = 0.0;
    }
  }

}

/* Print input matrices */
void print_inputs() {
  int row, col;

  if (N < 35) {
    printf("\nA =\n\t");
    for (row = 0; row < N; row++) {
      for (col = 0; col < N; col++) {
	    printf("%5.2f%s", A[row][col], (col < N-1) ? ", " : ";\n\t");
      }
    }
  }
}

void print_B() {
    int row, col;

    if (N < 35) {
        printf("\nB =\n\t");
        for (row = 0; row < N; row++) {
            for (col = 0; col < N; col++) {
                printf("%1.10f%s", B[row][col], (col < N-1) ? ", " : ";\n\t");
            }
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
  matrixNorm();

  /* Stop Clock */
  gettimeofday(&etstop, &tzdummy);
  etstop2 = times(&cputstop);
  printf("Stopped clock.\n");
  usecstart = (unsigned long long)etstart.tv_sec * 1000000 + etstart.tv_usec;
  usecstop = (unsigned long long)etstop.tv_sec * 1000000 + etstop.tv_usec;

  /* Display output */
  print_B();

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
/* Provided global variables are MAXN, N, A[][] and B[][],
 * defined in the beginning of this code.  B[][] is initialized to zeros.;
 */

#define BLOCK_SIZE 32

// http://stackoverflow.com/questions/20086047/cuda-matrix-example-block-size
void printError(cudaError_t err, char* string) {
    if(err != 0) {
        printf("CUDA ERROR: %s\n", cudaGetErrorString(err));
        printf("Description:  %s\n",string);
        getchar();
    }
}

/**
This function performs the partial sum of the given arrays
It is an improvement over the partial sum example from class
Inspired in the code found in https://gist.github.com/wh5a/4424992
The code there has been studied, as the comments indicate

The code had to be adapted to operate with arrays of different dimensions, 
as well as to operate adding columns instead of rows
*/
__global__ void partialSum(float * input, float * output, const int N) {

    // Load a segment of the input vector into shared memory
    __shared__ float partialSum[2 * BLOCK_SIZE * BLOCK_SIZE];

    // Position variables
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int ty = threadIdx.y;
    unsigned int tx = threadIdx.x;

    // Where does the calculation start for this iteration, based on the block X position
    unsigned int start = 2 * blockIdx.y * BLOCK_SIZE;

    // column modifier that we apply to partialSum[]
    unsigned int column = 2 * BLOCK_SIZE * tx;

    // Verify that we are inside the array, so CUDA won't throw errors
    if ( y >= N || x >= N )
      return;

    // If we are inside the input array, we transfer the value that we're going to sum up to the partial sum array
    if (start + ty < N)
       partialSum[ ty + column ] = input[ (start + ty)*MAXN + x ];
    else
       partialSum[ ty + column ] = 0;

    // The same for the last element of the block, the other value that we're going to sum up
    if (start + BLOCK_SIZE + ty < N)
       partialSum[BLOCK_SIZE + ty + column] = input[ (start + BLOCK_SIZE + ty)*MAXN + x ];
    else
       partialSum[BLOCK_SIZE + ty + column] = 0;  

    // Perform the partial sum
    for (unsigned int stride = BLOCK_SIZE; stride >= 1; stride >>= 1) {
       __syncthreads();
       if (ty < stride)
          partialSum[ty + column] += partialSum[ty+stride + column];
    }
    // After the loop, the partial sum is found in partialSum[0]
    // So we have to put it in the output array
    if (ty == 0)
       output[blockIdx.y*N + x] = partialSum[column];
}

/**
  This function behaves in the same way that partialSum but the input is modified every time using the mean parameter
  The function performed when reading the input is (A[i][j] - mean)^2
  By doing this, we integrate the step of that calculation with the sum of all columns
*/
__global__ void partialSumMeanDifferences(float * input, float * output, float * means, const int N) {

    // Load a segment of the input vector into shared memory
    __shared__ float partialSum[2 * BLOCK_SIZE * BLOCK_SIZE];

    // Position variables
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int ty = threadIdx.y;
    unsigned int tx = threadIdx.x;

    // Where does the calculation start for this iteration, based on the block X position
    unsigned int start = 2 * blockIdx.y * BLOCK_SIZE;

    // column modifier that we apply to partialSum[]
    unsigned int column = 2 * BLOCK_SIZE * tx;

    // Verify that we are inside the array, so CUDA won't throw errors
    if ( y >= N || x >= N )
      return;

    // If we are inside the input array, we transfer the value that we're going to sum up to the partial sum array
    if (start + ty < N)
       partialSum[ ty + column ] = powf(input[ (start + ty)*MAXN + x ] - means [ x ], 2);
    else
       partialSum[ ty + column ] = 0;

    // The same for the last element of the block, the other value that we're going to sum up
    if (start + BLOCK_SIZE + ty < N)
       partialSum[BLOCK_SIZE + ty + column] = powf(input[ (start + BLOCK_SIZE + ty)*MAXN + x ] - means [ x ], 2);
    else
       partialSum[BLOCK_SIZE + ty + column] = 0;  

    // Perform the partial sum
    for (unsigned int stride = BLOCK_SIZE; stride >= 1; stride >>= 1) {
       __syncthreads();
       if (ty < stride)
          partialSum[ty + column] += partialSum[ty+stride + column];
    }
    // After the loop, the partial sum is found in partialSum[0]
    // So we have to put it in the output array
    if (ty == 0)
       output[blockIdx.y*N + x] = partialSum[column];
}

/**
 This function performs the operation of normalizing 
 That is applying B[row][col] = (A[row][col] – mean) / standard_deviation
*/
__global__ void normalize(float * input, float * output, float * means, float * deviations, const int N) {

    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Verify that we are inside the array, so CUDA won't throw errors
    if ( y >= N || x >= N )
      return;

    if ( deviations [x] == 0 )
       output[ x + y*MAXN ] = 0;
    else
      output[ x + y*MAXN ] = ( input[ x + y*MAXN ] - means [x] ) / deviations [x];
}



void matrixNorm() {

  printf("Computing using CUDA.\n");
  
  // Size of input and output arrays (they are initialized with MAXN)
  int size = MAXN*MAXN*sizeof(float);

  // Size of the output of the partial sums algorithm
  int Nsums = ceil( ((float)N) / (BLOCK_SIZE<<1));
  int sizeSums = N*Nsums*sizeof(float);

  int row, col;

  float *d_sums, *d_A, *d_B;

  //Get user input into size;
  //float (*h_sums)[BLOCK_SIZE] = new float[N][BLOCK_SIZE];
  float *h_sums;
  h_sums = (float*)malloc(sizeSums);
  for (int i=0; i < Nsums; i++)
      for (int j=0; j < N; j++)
          h_sums[i*N + j] = 0;
      

  /*printf("MATRIX h_sums BEFORE\n\t");
  for (row = 0; row < Nsums; row++) {
      for (col = 0; col < N; col++) {
          printf("%1.1f%s", h_sums[row*N + col], (col < N-1) ? ", " : ";\n\t");
      }
  }*/
/*
  for (int i=0; i < N; i++)
      for (int j=0; j < N; j++) {
        if ( i == 0 )
          A[i][j] = j;
        else
          A[i][j] = 1;
      }*/
/*
  printf("MATRIX A BEFORE\n\t");
  for (row = 0; row < N; row++) {
      for (col = 0; col < N; col++) {
          printf("%1.1f%s", A[row][col], (col < N-1) ? ", " : ";\n\t");
      }
  }*/
  

  // Allocate space for variables
  printError( cudaMalloc( (void**)&d_A, size ) , "Error allocating memory for d_A before partialSum()");
  printError( cudaMalloc( (void**)&d_sums, sizeSums ) , "Error allocating memory for d_sums before partialSum()");

  // Copy the matrix A and the matrix that will contain the output of the partial sums algorithm
  printError( cudaMemcpy( d_A, A, size, cudaMemcpyHostToDevice) , "Error copying from A to d_A before partialSum()");
  printError( cudaMemcpy( d_sums, h_sums, sizeSums, cudaMemcpyHostToDevice ) , "Error copying from h_sums to d_sums before partialSum()");

  
  int gridSize = ceil(((float)N)/BLOCK_SIZE);
  dim3 dimBlock( BLOCK_SIZE, BLOCK_SIZE );
  dim3 dimGrid( gridSize, gridSize);

  // 
  // Use reduction with partial sum algorithm to create partial sums of column values with complexity O(log(N))
  //
  partialSum<<< dimGrid, dimBlock>>> (d_A, d_sums, N);

  cudaError_t  err = cudaGetLastError();
  if ( cudaSuccess != err )
    printError( err, "Error in partialSum()" );

  printError( cudaMemcpy( h_sums, d_sums, sizeSums, cudaMemcpyDeviceToHost ) , "Error copying from d_sums to h_sums after partialSum()");

  // 
  // Add reducted means sequentially. After that, divide by N, total number of elements in a column
  //
  float *h_means;
  h_means = (float*)malloc( N*sizeof(float) );
  for ( int i = 0; i < N; i++ )
    h_means[i] = 0;

  for ( int i = 0; i < Nsums; i++ )
    for ( int j = 0; j < N; j++ )
      h_means[j] += h_sums[i*N+j];

  // Divide between number of elements
  for ( int i = 0; i < N; i++ )
    h_means[i] /= N;

/*
  printf("MATRIX h_means AFTER\n\t");
  for ( int i = 0; i < N; i++ )
    printf("%1.2f%s", h_means[i], (i < N-1) ? ", " : ";\n\t");
*/

  // 
  // Transfer means to CUDA
  //
  float *d_means;
  printError( cudaMalloc( (void**)&d_means, N*sizeof(float) ) , "Error allocating memory for d_means before  partialSumMeanDifferences()");
  printError( cudaMemcpy( d_means, h_means, N*sizeof(float), cudaMemcpyHostToDevice) , "Error copying from h_means to d_means before  partialSumMeanDifferences()");

  // 
  // Calculate the value of (A[i][j] - mean)^2
  // Add all the operands (A[i][j] - mean)^2 in each column
  // It is the same operation of adding all values in columns that we did in the step of calculating the mean
  //
  partialSumMeanDifferences<<< dimGrid, dimBlock>>> (d_A, d_sums, d_means, N);

  err = cudaGetLastError();
  if ( cudaSuccess != err )
    printError( err, "Error in partialSumMeanDifferences()" );

  //printError( cudaMemcpy( A, d_A, size, cudaMemcpyDeviceToHost ) , "Error copying from d_A to A");

  printError( cudaMemcpy( h_sums, d_sums, sizeSums, cudaMemcpyDeviceToHost ) , "Error copying from d_sums to h_sums after partialSumMeanDifferences()");
  printError( cudaFree(d_sums) , "Error freeing memory of d_sums after partialSumMeanDifferences()");

  // 
  // Add reducted means sequentially. After that, divide by N and calculate square root
  //
  for ( int i = 0; i < N; i++ )
    h_means[i] = 0;

  for ( int i = 0; i < Nsums; i++ )
    for ( int j = 0; j < N; j++ )
      h_means[j] += h_sums[i*N+j];

  // Divide between number of elements
  for ( int i = 0; i < N; i++ )
    h_means[i] = powf(h_means[i]/N, 0.5f);

  /*printf("MATRIX h_means AFTER QUADRATIC ADDING\n\t");
  for ( int i = 0; i < N; i++ )
    printf("%1.2f%s", h_means[i], (i < N-1) ? ", " : ";\n\t");
  */

  float *d_sigmas;
  printError( cudaMalloc( (void**)&d_sigmas, N*sizeof(float) ) , "Error allocating memory for d_sigmas before normalize()");
  printError( cudaMemcpy( d_sigmas, h_means, N*sizeof(float), cudaMemcpyHostToDevice) , "Error copying from h_means to d_sigmas before normalize()");
  
  // 
  // Apply the formula to normalize
  // B[row][col] = (A[row][col] – mean) / standard_deviation
  //

  printError( cudaMalloc( (void**)&d_B, size ) , "Error allocating memory for d_B before normalize()");

  normalize<<< dimGrid, dimBlock>>> (d_A, d_B, d_means, d_sigmas, N);

  err = cudaGetLastError();
  if ( cudaSuccess != err )
    printError( err, "Error in normalize()" );

  printError( cudaMemcpy( B, d_B, size, cudaMemcpyDeviceToHost ) , "Error copying memory from d_B to B after normalize()");

  printError( cudaFree(d_A) , "Error freeing memory of d_A after normalize()");
  printError( cudaFree(d_B) , "Error freeing memory ofd_B d_A after normalize()");
  printError( cudaFree(d_means) , "Error freeing memory of d_A after normalize()");
  printError( cudaFree(d_sigmas) , "Error d_means memory of d_sigmas after normalize()");
  
  free ( h_sums );
  free ( h_means );
}


