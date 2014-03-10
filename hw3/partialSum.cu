

#include <stdio.h>
#include <iostream>

using namespace std;


__global__ void 
partialSum(float *partialSum) {

	unsigned int t = threadId.x;
	for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
	__syncthreads();
	if (t < stride)
		partialSum[t] += partialSum[t+stride];
	}
}



int
main()
{   
	int size = 12;
	int num_bytes = size*sizeof(float);

	float *d_a, *h_a;

	h_a = (float*)malloc(num_bytes);

	for (int i=0; i < N; i++){   
	    h_a[i]=0;
	}

	h_a[0]=1; h_a[1]=3;h_a[2]=2;

	cudaMalloc( (void**)&d_a, num_bytes );

	cudaMemcpy( d_a, h_a, num_bytes, cudaMemcpyHostToDevice);


	partialSum<<< ceil(size / 4), 4>>> (d_a);

	cudaMemcpy( h_a, d_a, num_bytes, cudaMemcpyDeviceToHost );

	cudaFree(d_a);

	printf("MATRIX A\n\t");
    int i;
	for (i = 0; i < size; i++) {
      cout << "h_array[" << (i) + j << "]=" << h_array[i] << endl;
    } 
    free(h_a);
}
