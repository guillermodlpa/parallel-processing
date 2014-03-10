

#include <stdio.h>
#include <iostream>

using namespace std;


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



int
main()
{   
	int N = 17;
	int num_bytes = N*sizeof(float);

	float *d_a, *h_a;

	h_a = (float*)malloc(num_bytes);

	for (int i=0; i < N; i++){   
	    h_a[i]=0;
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
	cudaMemcpy( d_a, h_a, num_bytes, cudaMemcpyHostToDevice);
	cout << "Size of grid " << ceil(  ((float)N)/blocksize) << endl;
	int blocksize = 8;
	dim3 dimBlock( blocksize, 1 );
	dim3 dimGrid( ceil(  ((float)N)/blocksize), 1 );

	partialSum<<< dimGrid, blocksize>>> (d_a, N);

	cudaMemcpy( h_a, d_a, num_bytes, cudaMemcpyDeviceToHost );

	cudaFree(d_a);

	printf("MATRIX AFTER\n\t");
	for (i = 0; i < N; i++) {
      cout << "h_a[" << i << "]=" << h_a[i] << endl;
    } 
    free(h_a);
}
