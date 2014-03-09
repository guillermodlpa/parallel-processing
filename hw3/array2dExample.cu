
#include <stdio.h>
#include <iostream>

using namespace std;

__global__ void 
reduce(float *g, float *o, const int dimx, const int dimy) {

	//extern __shared__ float sdata[];

	//unsigned int tid_x = threadIdx.x;
	//unsigned int tid_y = threadIdx.y;

	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y * blockIdx.y + threadIdx.y; 

	if (i >= dimx || j >= dimy)
	    return;

	o[i+j] = g[i+j] + 1;
}




int
main()
{   
	int dimx = 32;
	int dimy = 16;
	int num_bytes = dimx*dimy*sizeof(float);

	float *d_a, *h_a, // device and host pointers
	            *d_o=0, *h_o=0;

	h_a = (float*)malloc(num_bytes);
	h_o = (float*)malloc(num_bytes);

	for (int i=0; i < dimx; i++){   
	    for (int j=0; j < dimy; j++){
	        h_a[i*dimy + j] = 1;
	    }
	}
	for (int i=0; i < dimx; i++){   
	    for (int j=0; j < dimy; j++){
	        h_o[i*dimy + j] = 0;
	    }
	}

	cudaMalloc( (void**)&d_a, num_bytes );
	cudaMalloc( (void**)&d_o, sizeof(int) );

	cudaMemcpy( d_a, h_a, num_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy( d_o, h_o, sizeof(int), cudaMemcpyHostToDevice); 

	dim3 grid, block;
	block.x = 4;
	block.y = 4;
	grid.x = dimx / block.x;
	grid.y = dimy / block.y;

	reduce<<<grid, block>>> (d_a, d_o, dimx, dimy);

	std::cout << block.x << " " << block.y << std::endl;
	std::cout << grid.x << " " << grid.y << std::endl;
	std::cout << dimx <<  " " << dimy << " " << dimx*dimy << std::endl;

	cudaMemcpy( h_a, d_a, num_bytes, cudaMemcpyDeviceToHost );
	cudaMemcpy( h_o, d_o, num_bytes, cudaMemcpyDeviceToHost );


	printf("MATRIX A\n\t");
    int row, col;
	for (row = 0; row < dimx; row++) {
      for (col = 0; col < dimy; col++) {
          printf("%1.0f%s", h_a[(row*dimy+col)], (col < dimy-1) ? ", " : ";\n\t");
      }
    } 

	printf("MATRIX O\n\t");
  	for (row = 0; row < dimx; row++) {
      for (col = 0; col < dimy; col++) {
          printf("%1.0f%s", h_o[(row*dimy+col)], (col < dimy-1) ? ", " : ";\n\t");
      }
    } 

    cudaFree(d_a);
	cudaFree(d_o);
    free(h_a);
    free(h_o);
}
