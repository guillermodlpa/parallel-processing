// This is the REAL "hello world" for CUDA!
// It takes the string "Hello ", prints it, then passes it to CUDA with an array
// of offsets. Then the offsets are added in parallel to produce the string "World!"
// By Ingemar Ragnemalm 2010
// http://computer-graphics.se/hello-world-for-cuda.html
 
#include <stdio.h>
 
const int N = 16; 
const int blocksize = 16; 
 
__global__ 
void hello(float **dA) 
{
	dA[threadIdx.x][threadIdx.y] = threadIdx.x;
}
 
int main()
{
	char a[N] = "Hello \0\0\0\0\0\0";
	int b[N] = {15, 10, 6, 0, -11, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

	float A[N][N];
	float dA[N][N];
 
	const int asize = N*N*sizeof(float);
 
 
	cudaMalloc( (void**)&dA, asize ); 
	cudaMemcpy( dA, A, asize, cudaMemcpyHostToDevice ); 
	
	dim3 dimBlock( blocksize, blocksize );
	dim3 dimGrid( 1, 1 );
	hello<<<dimGrid, dimBlock>>>(&dA);
	cudaMemcpy( A, dA, asize, cudaMemcpyDeviceToHost ); 
	cudaFree( dA );

	int row, col;
	for (row = 0; row < N; row++) {
        for (col = 0; col < N; col++) {
            printf("%1.10f%s", A[row][col], (col < N-1) ? ", " : ";\n\t");
        }
    }
	
	return EXIT_SUCCESS;
}