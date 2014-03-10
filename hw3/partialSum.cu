

#include <stdio.h>
#include <iostream>

using namespace std;


__global__ void 
partialSum(float *partialSum, const int N) {

	unsigned int t = threadIdx.x;

	for (unsigned int stride = blockDim.x; stride > 0; stride >>= 1) {

		__syncthreads();
		if (t < stride) {
			int i = 1;
			do {
				partialSum[t] += partialSum[t+stride*i];
				i++;
			} while ( t+stride*i < N );

		}
			
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

	h_a[0]=1; h_a[1]=3;h_a[2]=2;h_a[3]=1; h_a[4]=3;h_a[5]=2;h_a[6]=0.5;
	h_a[7]=1; h_a[8]=3;h_a[9]=2;h_a[10]=1; h_a[11]=3;

	printf("MATRIX BEFORE\n\t");
    int i;
	for (i = 0; i < N; i++) {
      cout << "h_a[" << i << "]=" << h_a[i] << endl;
    } 

	cudaMalloc( (void**)&d_a, num_bytes );
	cudaMemcpy( d_a, h_a, num_bytes, cudaMemcpyHostToDevice);


	partialSum<<< ceil(N / 4), 4>>> (d_a, N);

	cudaMemcpy( h_a, d_a, num_bytes, cudaMemcpyDeviceToHost );

	cudaFree(d_a);

	printf("MATRIX AFTER\n\t");
	for (i = 0; i < N; i++) {
      cout << "h_a[" << i << "]=" << h_a[i] << endl;
    } 
    free(h_a);
}
