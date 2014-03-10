

#include <stdio.h>
#include <iostream>

using namespace std;

#define BLOCK_SIZE 8

__global__ void 
partialSum(float *partialSum, const int N) {

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if ( x >= N )
		return;

	unsigned int t = threadIdx.x;

	for (unsigned int stride = N/2; stride > 0; stride >>= 1) {
		if (t < stride) {
			partialSum[t] += partialSum[t+stride];
		}
		__syncthreads();
	}
}

__global__ void total(float * input, float * output, int len) {
    //@@ Load a segment of the input vector into shared memory
    __shared__ float partialSum[2 * BLOCK_SIZE];
    unsigned int t = threadIdx.x, start = 2 * blockIdx.x * BLOCK_SIZE;
    if (start + t < len)
       partialSum[t] = input[start + t];
    else
       partialSum[t] = 0;
    if (start + BLOCK_SIZE + t < len)
       partialSum[BLOCK_SIZE + t] = input[start + BLOCK_SIZE + t];
    else
       partialSum[BLOCK_SIZE + t] = 0;
    //@@ Traverse the reduction tree
    for (unsigned int stride = BLOCK_SIZE; stride >= 1; stride >>= 1) {
       __syncthreads();
       if (t < stride)
          partialSum[t] += partialSum[t+stride];
    }
    //@@ Write the computed sum of the block to the output vector at the 
    //@@ correct index
    if (t == 0)
       output[blockIdx.x] = partialSum[0];
}


int
main()
{   
	int N = 17;
	int num_bytes = N*sizeof(float);

	float *d_a, *h_a, *h_o, *d_o;

	h_a = (float*)malloc(num_bytes);
	h_o = (float*)malloc(num_bytes);

	for (int i=0; i < N; i++){   
	    h_a[i]=0; h_o[i]=0;
	}

	h_a[0]=1; 
	h_a[1]=1;
	h_a[2]=1;
	h_a[3]=1;
	h_a[4]=1;
	h_a[5]=1;
	h_a[6]=1;
	h_a[7]=1;
	h_a[8]=1;
	h_a[9]=1;
	h_a[10]=1;
	h_a[11]=1;
	h_a[12]=1;
	h_a[13]=1;
	h_a[14]=1;
	h_a[15]=1;
	h_a[16]=1;

	printf("MATRIX BEFORE\n\t");
    int i;
	for (i = 0; i < N; i++) {
      cout << "h_a[" << i << "]=" << h_a[i] << endl;
    } 

	cudaMalloc( (void**)&d_a, num_bytes );
	cudaMalloc( (void**)&d_o, num_bytes );
	cudaMemcpy( d_a, h_a, num_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy( d_o, h_o, num_bytes, cudaMemcpyHostToDevice);

	dim3 dimBlock( BLOCK_SIZE, 1 );
	dim3 dimGrid( ceil(  ((float)N)/BLOCK_SIZE), 1 );

	total<<< dimGrid, BLOCK_SIZE>>> (d_a, d_o, N);

	cudaMemcpy( h_a, d_a, num_bytes, cudaMemcpyDeviceToHost );
	cudaMemcpy( h_o, d_o, num_bytes, cudaMemcpyDeviceToHost );

	cudaFree(d_a);
	cudaFree(d_o);

	printf("MATRIX AFTER\n\t");
	for (i = 0; i < N; i++) {
      cout << "h_o[" << i << "]=" << h_o[i] << endl;
    } 
    free(h_a);
    free(h_o);
}
