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


/**
This function performs the partial sum of the given arrays
It is an improvement over the partial sum example from class
Inspired in the code found in https://gist.github.com/wh5a/4424992
The code there has been studied, as the comments indicate
*/
__global__ void 
partialSum(float *input, float *output, const int N, const int Noutput) {

  // Load a segment of the input vector into shared memory
  // This is because the entire array might be too big and is stored into the global memory
    __shared__ float partialSum[2* BLOCK_SIZE*BLOCK_SIZE];

    // Position in the input array
    unsigned int t = threadIdx.x;

    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int ty = threadIdx.y;


    if ( y >= N )
      return;

    // Start is the beining of the current calculations
    // If blockIdx is not 0, then the result will go to the blockIdx position of the output array
    unsigned int start = 2 * blockIdx.x * BLOCK_SIZE;

    // If we are inside the input array, we transfer the value that we're going to sum up to the partial sum array
    if (start + t < N)
       //partialSum[t+ty*BLOCK_SIZE] = input[start + t +y*N];
      partialSum[t + ty*2*BLOCK_SIZE] = input[start + t + y*N];
    else
       //partialSum[t+ty*BLOCK_SIZE] = 0;
      partialSum[t + ty*2*BLOCK_SIZE] = 0;
   
    // The same for the last element of the block, the other value that we're going to sum up
    if (start + BLOCK_SIZE + t < N)
       //partialSum[BLOCK_SIZE + t+ty*BLOCK_SIZE] = input[start + BLOCK_SIZE + t +y*N];
      partialSum[BLOCK_SIZE + t + ty*2*BLOCK_SIZE] = input[start + BLOCK_SIZE + t + y*N];
    else
       //partialSum[BLOCK_SIZE + t+y*2*BLOCK_SIZE] = 0;
      partialSum[BLOCK_SIZE + t + ty*2*BLOCK_SIZE] = 0;
   
    // Perform the partial sum
    for (unsigned int stride = BLOCK_SIZE; stride >= 1; stride >>= 1) {
       __syncthreads();
       if (t < stride)
          //partialSum[t+ty*BLOCK_SIZE] += partialSum[t+stride+ty*BLOCK_SIZE];
          partialSum[t + ty*2*BLOCK_SIZE] += partialSum[t+stride + ty*2*BLOCK_SIZE];
    }

    // After the loop, the partial sum is found in partialSum[0]
    // So we have to put it in the output array
    if (t == 0)
       //output[blockIdx.x + y*Noutput] += partialSum[0+ty*BLOCK_SIZE];
      output[blockIdx.x + y*Noutput] = partialSum[ty*2*BLOCK_SIZE];
}



void matrixNorm() {

  printf("Computing using CUDA.\n");

  // CALCULATING MEAN
  int size = N*N*sizeof(float);
  int Nmeans = ceil( ((float)N) / (BLOCK_SIZE<<1));
  int sizeMeans = N*Nmeans*sizeof(float);
  int row, col;

  float *d_means, *d_A, *d_B, *h_means;

  h_means = (float*)malloc(sizeMeans);

  for (int i=0; i < N; i++)
      for (int j=0; j < N; j++)
         A[i]j]=i+1;

  for (int i=0; i < Nmeans; i++)
      for (int j=0; j < N; j++)
         h_means[i*Nmeans+j]=0;

       printf("MATRIX BEFORE\n\t");
  
  for (row = 0; row < Nmeans; row++) {
      for (col = 0; col < N; col++) {
          printf("%1.1f%s", h_means[row +N*col], (col < N-1) ? ", " : ";\n\t");
      }
  }

  cudaMalloc( (void**)&d_A, size );
  cudaMalloc( (void**)&d_B, size );
  cudaMalloc( (void**)&d_means, sizeMeans );

  cudaMemcpy( d_A, A, size, cudaMemcpyHostToDevice);
  cudaMemcpy( d_means, h_means, sizeMeans, cudaMemcpyHostToDevice);

  dim3 dimBlock( BLOCK_SIZE, BLOCK_SIZE );
  dim3 dimGrid( ceil(((float)N)/BLOCK_SIZE), ceil(((float)N)/BLOCK_SIZE) );

  partialSum<<< dimGrid, dimBlock>>> (d_A, d_means, N, Nmeans);

  cudaMemcpy( h_means, d_means, sizeMeans, cudaMemcpyDeviceToHost );

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_means);

  printf("MATRIX AFTER\n\t");
  
  for (row = 0; row < Nmeans; row++)
    for (col=0; col < N; col++)
      printf("%1.1f%s", h_means[row+col*Nmeans], (col < N-1) ? ", " : ";\n\t");

}
/*
void matrixNorm() {
  int row, col; 
  float mu, sigma; // Mean and Standard Deviation

  printf("Computing using CUDA.\n");

    for (col=0; col < N; col++) {
        mu = 0.0;
        for (row=0; row < N; row++)
            mu += A[row][col];
        mu /= (float) N;
        sigma = 0.0;
        for (row=0; row < N; row++)
            sigma += powf(A[row][col] - mu, 2.0);
        sigma /= (float) N;
        sigma = sqrt(sigma);
        //printf("Mean eq %g.Sigma eq %g.\n", mu, sigma);
        for (row=0; row < N; row++) {
            if (sigma == 0.0)
                B[row][col] = 0.0;
            else
                B[row][col] = (A[row][col] - mu) / sigma;
        }
    }

}*/


// http://stackoverflow.com/questions/20086047/cuda-matrix-example-block-size
void printError(cudaError_t err) {
    if(err != 0) {
        printf("CUDA ERROR: %s\n", cudaGetErrorString(err));
        getchar();
    }
}

