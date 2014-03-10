

#include <stdio.h>
#include <iostream>

using namespace std;


__global__ void 
partialSum(float *partialSum) {

	unsigned int t = threadIdx.x;
	for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
	__syncthreads();
	if (t < stride)
		partialSum[t] += partialSum[t+stride];
	}
}



int
main()
{   
	int N = 12;
	int num_bytes = N*sizeof(float);

	float *d_a, *h_a;

	h_a = (float*)malloc(num_bytes);

	for (int i=0; i < N; i++){   
	    h_a[i]=0;
	}

	h_a[0]=1; h_a[1]=3;h_a[2]=2;

	printf("MATRIX BEFORE\n\t");
    int i;
	for (i = 0; i < N; i++) {
      cout << "h_a[" << i << "]=" << h_a[i] << endl;
    } 

	cudaMalloc( (void**)&d_a, num_bytes );
	cudaMemcpy( d_a, h_a, num_bytes, cudaMemcpyHostToDevice);


	partialSum<<< ceil(N / 4), 4>>> (d_a);

	cudaMemcpy( h_a, d_a, num_bytes, cudaMemcpyDeviceToHost );

	cudaFree(d_a);

	printf("MATRIX AFTER\n\t");
	for (i = 0; i < N; i++) {
      cout << "h_a[" << i << "]=" << h_a[i] << endl;
    } 
    free(h_a);
}
