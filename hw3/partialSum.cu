

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
partialSum(float *input, float *output, const int N) {

	// Load a segment of the input vector into shared memory
	// This is because the entire array might be too big and is stored into the global memory
    __shared__ float partialSum[2 * BLOCK_SIZE];

    // Position in the input array
    unsigned int t = threadIdx.x;

    // Start is the beining of the current calculations
    // If blockIdx is not 0, then the result will go to the blockIdx position of the output array
    unsigned int start = 2 * blockIdx.x * BLOCK_SIZE;

    // If we are inside the input array, we transfer the value that we're going to sum up to the partial sum array
    if (start + t < N)
       partialSum[t] = input[start + t];
    else
       partialSum[t] = 0;
   
    // The same for the last element of the block, the other value that we're going to sum up
    if (start + BLOCK_SIZE + t < N)
       partialSum[BLOCK_SIZE + t] = input[start + BLOCK_SIZE + t];
    else
       partialSum[BLOCK_SIZE + t] = 0;
   
    // Perform the partial sum
    for (unsigned int stride = BLOCK_SIZE; stride >= 1; stride >>= 1) {
       __syncthreads();
       if (t < stride)
          partialSum[t] += partialSum[t+stride];
    }

    // After the loop, the partial sum is found in partialSum[0]
    // So we have to put it in the output array
    if (t == 0)
       output[blockIdx.x] = partialSum[0];
}


int
main()
{   
	int N = 25;
	int sizeInput = N*sizeof(float);
	int Noutput = ceil( ((float)N) / (BLOCK_SIZE<<1));
	int sizeOutput = Noutput*sizeof(float);

	float *d_a, *h_a, *h_o, *d_o;

	h_a = (float*)malloc(sizeInput);
	h_o = (float*)malloc(sizeOutput);

	for (int i=0; i < N; i++){   
	    h_a[i]=1;
	}

	printf("MATRIX BEFORE\n\t");
    int i;
	for (i = 0; i < N; i++) {
      cout << "h_a[" << i << "]=" << h_a[i] << endl;
    } 

	cudaMalloc( (void**)&d_a, sizeInput );
	cudaMalloc( (void**)&d_o, sizeOutput );
	cudaMemcpy( d_a, h_a, sizeInput, cudaMemcpyHostToDevice);
	cudaMemcpy( d_o, h_o, sizeOutput, cudaMemcpyHostToDevice);

	dim3 dimBlock( BLOCK_SIZE, 1 );
	dim3 dimGrid( ceil(  ((float)N)/BLOCK_SIZE), 1 );

	partialSum<<< dimGrid, BLOCK_SIZE>>> (d_a, d_o, N);

	cudaMemcpy( h_a, d_a, sizeInput, cudaMemcpyDeviceToHost );
	cudaMemcpy( h_o, d_o, sizeOutput, cudaMemcpyDeviceToHost );

	cudaFree(d_a);
	cudaFree(d_o);

	printf("MATRIX AFTER\n\t");
	for (i = 0; i < Noutput; i++) {
      cout << "h_o[" << i << "]=" << h_o[i] << endl;
    } 
    free(h_a);
    free(h_o);
}
