

#include <stdio.h>
#include <iostream>

using namespace std;

#define BLOCK_SIZE 8


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

    // Start is the beining of the current calculations
    // If blockIdx is not 0, then the result will go to the blockIdx position of the output array
    unsigned int start = 2 * blockIdx.x * BLOCK_SIZE;

    // If we are inside the input array, we transfer the value that we're going to sum up to the partial sum array
    if (start + t < N)
       partialSum[t+ty*BLOCK_SIZE] = input[start + t +y*N];
    else
       partialSum[t+ty*BLOCK_SIZE] = 0;
   
    // The same for the last element of the block, the other value that we're going to sum up
    if (start + BLOCK_SIZE + t < N)
       partialSum[BLOCK_SIZE + t+ty*BLOCK_SIZE] = input[start + BLOCK_SIZE + t +y*N];
    else
       partialSum[BLOCK_SIZE + t+y*2*BLOCK_SIZE] = 0;
   
    // Perform the partial sum
    for (unsigned int stride = BLOCK_SIZE; stride >= 1; stride >>= 1) {
       __syncthreads();
       if (t < stride)
          partialSum[t+ty*BLOCK_SIZE] += partialSum[t+stride+ty*BLOCK_SIZE];
    }

    // After the loop, the partial sum is found in partialSum[0]
    // So we have to put it in the output array
    if (t == 0)
       output[blockIdx.x + y*Noutput] = partialSum[0+ty*BLOCK_SIZE];
}


int
main()
{   
	int N = 25;
	int sizeInput = N*N*sizeof(float);
	int Noutput = ceil( ((float)N) / (BLOCK_SIZE<<1));
	int sizeOutput = N*Noutput*sizeof(float);

	float *d_a, *h_a, *h_o, *d_o;

	h_a = (float*)malloc(sizeInput);
	h_o = (float*)malloc(sizeOutput);

	for (int i=0; i < N; i++)
      for (int j=0; j < N; j++)
	       h_a[i*N+j]=1;
  for (int i=0; i < Noutput; i++)
      for (int j=0; j < N; j++)
         h_o[i*Noutput+j]=0;
	

	printf("MATRIX BEFORE\n\t");
  int row, col;
	for (row = 0; row < N; row++)
    for (col=0; col < N; col++)
      printf("%1.1f%s", h_a[row +N*col], (col < N-1) ? ", " : ";\n\t");

	cudaMalloc( (void**)&d_a, sizeInput );
	cudaMalloc( (void**)&d_o, sizeOutput );
	cudaMemcpy( d_a, h_a, sizeInput, cudaMemcpyHostToDevice);
	cudaMemcpy( d_o, h_o, sizeOutput, cudaMemcpyHostToDevice);

	dim3 dimBlock( BLOCK_SIZE, BLOCK_SIZE );
	dim3 dimGrid( ceil(  ((float)N)/BLOCK_SIZE), ceil(  ((float)N)/BLOCK_SIZE) );

	partialSum<<< dimGrid, dimBlock>>> (d_a, d_o, N, Noutput);

	cudaMemcpy( h_a, d_a, sizeInput, cudaMemcpyDeviceToHost );
	cudaMemcpy( h_o, d_o, sizeOutput, cudaMemcpyDeviceToHost );

	cudaFree(d_a);
	cudaFree(d_o);

	printf("MATRIX AFTER\n\t");
	for (row = 0; row < Noutput; row++)
    for (col=0; col < N; col++)
      printf("%1.1f%s", h_o[row +N*col], (col < N-1) ? ", " : ";\n\t");
    free(h_a);
    free(h_o);
}
