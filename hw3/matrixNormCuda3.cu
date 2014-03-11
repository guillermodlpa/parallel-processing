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

  if (N < 10) {
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

    if (N < 10) {
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

#define BLOCK_SIZE 4

// http://stackoverflow.com/questions/20086047/cuda-matrix-example-block-size
void printError(cudaError_t err) {
    if(err != 0) {
        printf("CUDA ERROR: %s\n", cudaGetErrorString(err));
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


__global__ void partialSum2(float * input, float * output, float * means, const int N) {

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
       partialSum[ ty + column ] = powf(input[ (start + ty)*MAXN + x ] - means [ x ], 2)
    else
       partialSum[ ty + column ] = 0;

    // The same for the last element of the block, the other value that we're going to sum up
    if (start + BLOCK_SIZE + ty < N)
       partialSum[BLOCK_SIZE + ty + column] = powf(input[ (start + BLOCK_SIZE + ty)*MAXN + x ] - means [ x ], 2)
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



__global__ void calculateQuadratic(float * input, float * means, const int N) {

    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Verify that we are inside the array, so CUDA won't throw errors
    if ( y >= N || x >= N )
      return;

    input[ x + y*MAXN ] = powf(input[ x + y*MAXN ] - means [ x ], 2);
}



void matrixNorm() {

  printf("Computing using CUDA.\n");

  // CALCULATING MEAN
  int size = MAXN*MAXN*sizeof(float);
  int Nsums = ceil( ((float)N) / (BLOCK_SIZE<<1));
  int sizeSums = N*Nsums*sizeof(float);
/*
  int Nsums2 = ceil( ((float)Nsums) / (BLOCK_SIZE<<1));
  int sizeSums2 = N*Nsums2*sizeof(float);
*/
  int row, col;

  float *d_sums, *d_A, *d_B;

  //Get user input into size;
  //float (*h_sums)[BLOCK_SIZE] = new float[N][BLOCK_SIZE];
  float *h_sums;
  h_sums = (float*)malloc(sizeSums);
  for (int i=0; i < Nsums; i++)
      for (int j=0; j < N; j++)
          h_sums[i*N + j] = -1;
      

  printf("MATRIX h_sums BEFORE\n\t");
  for (row = 0; row < Nsums; row++) {
      for (col = 0; col < N; col++) {
          printf("%1.1f%s", h_sums[row*N + col], (col < N-1) ? ", " : ";\n\t");
      }
  }

  for (int i=0; i < N; i++)
      for (int j=0; j < N; j++) {
        if ( i == 0 )
          A[i][j] = j;
        else
          A[i][j] = 1;
      }

  printf("MATRIX A BEFORE\n\t");
  for (row = 0; row < N; row++) {
      for (col = 0; col < N; col++) {
          printf("%1.1f%s", A[row][col], (col < N-1) ? ", " : ";\n\t");
      }
  }
  

  // Allocagte space for variables
  printError( cudaMalloc( (void**)&d_A, size ) );
  printError( cudaMalloc( (void**)&d_B, size ) );
  printError( cudaMalloc( (void**)&d_sums, sizeSums ) );

  printError( cudaMemcpy( d_A, A, size, cudaMemcpyHostToDevice) );
  printError( cudaMemcpy( d_sums, h_sums, sizeSums, cudaMemcpyHostToDevice ));

  int gridSize = ceil(((float)N)/BLOCK_SIZE);

  dim3 dimBlock( BLOCK_SIZE, BLOCK_SIZE );
  dim3 dimGrid( gridSize, gridSize);

  // 
  // Use reduction with partial sum algorithm to create partial sums of column values with complexity O(log(N))
  //
  partialSum<<< dimGrid, dimBlock>>> (d_A, d_sums, N);

  printError( cudaMemcpy( h_sums, d_sums, sizeSums, cudaMemcpyDeviceToHost ) );

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

  printf("MATRIX h_means AFTER\n\t");
  for ( int i = 0; i < N; i++ )
    printf("%1.2f%s", h_means[i], (i < N-1) ? ", " : ";\n\t");


  // 
  // Transfer means to CUDA
  //
  float *d_means;
  printError( cudaMalloc( (void**)&d_means, N*sizeof(float) ) );
  printError( cudaMemcpy( d_means, h_means, N*sizeof(float), cudaMemcpyHostToDevice) );

  // 
  // Calculate the value of (A[i][j] - mean)^2
  // We put the value in the input array because we won't need the original a[i][j] values anymore
  //
  /*
  calculateQuadratic<<< dimGrid, dimBlock>>> (d_A, d_means, N);

  printf("MATRIX A AFTER QUADRATIC CALCULATION\n\t");
  for (row = 0; row < N; row++) {
      for (col = 0; col < N; col++) {
          printf("%1.3f%s", A[row][col], (col < N-1) ? ", " : ";\n\t");
      }
  }*/

  //
  // Add all the operands (A[i][j] - mean)^2 in each column
  // It is the same operation of adding all values in columns that we did in the step of calculating the mean
  //
  partialSum2<<< dimGrid, dimBlock>>> (d_A, d_sums, d_means, N);

  printError( cudaMemcpy( A, d_A, size, cudaMemcpyDeviceToHost ) );

  printError( cudaMemcpy( h_sums, d_sums, sizeSums, cudaMemcpyDeviceToHost ) );

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

  printf("MATRIX h_means AFTER QUADRATIC ADDING\n\t");
  for ( int i = 0; i < N; i++ )
    printf("%1.2f%s", h_means[i], (i < N-1) ? ", " : ";\n\t");

  
  // 
  // Apply the formula to normalize
  // B[row][col] = (A[row][col] – mean) / standard_deviation
  //


  printError( cudaFree(d_A) );
  printError( cudaFree(d_B) );
  printError( cudaFree(d_means) );
  
}


