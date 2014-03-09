// This is the REAL "hello world" for CUDA!
// It takes the string "Hello ", prints it, then passes it to CUDA with an array
// of offsets. Then the offsets are added in parallel to produce the string "World!"
// By Ingemar Ragnemalm 2010
// http://computer-graphics.se/hello-world-for-cuda.html
 
#include <stdio.h>
 
const int N = 16; 
const int blocksize = 16; 
 
__global__ 
void hello(float* dA) 
{
	int y = blockIdx.y * blockDim.y + threadIdx.y;
  	int x = blockIdx.x * blockDim.x + threadIdx.x;
	dA[x][y] = 1.0;
}
 
int main()
{

	float* A = new float [N][N];
	float* dA;

	int row, col;
	for (col = 0; col < N; col++) {
	    for (row = 0; row < N; row++) {
	      A[row][col] = 0.0;
	    }
	  }
 
	const int asize = N*N*sizeof(float);
 
 
	cudaMalloc( (void**)&dA, asize ); 
	cudaMemcpy( dA, A, asize, cudaMemcpyHostToDevice ); 
	
	dim3 dimBlock( blocksize, blocksize );
	dim3 dimGrid( blocksize, blocksize );
	hello<<<dimGrid, dimBlock>>>(dA);
	cudaMemcpy( A, dA, asize, cudaMemcpyDeviceToHost ); 
	cudaFree( dA );

	
	for (row = 0; row < N; row++) {
        for (col = 0; col < N; col++) {
            printf("%1.0f%s", A[row][col], (col < N-1) ? ", " : ";\n\t");
        }
    }
	
	return EXIT_SUCCESS;
}