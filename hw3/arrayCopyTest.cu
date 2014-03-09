#include <stdio.h>

// kernel which copies data from d_array to destinationArray
__global__ void CopyData(float* d_array, 
                                   float* destinationArray, 
                                   size_t pitch, 
                                   int columnCount, 
                                   int rowCount)
{
  for (int row = 0; row < rowCount; row++) 
  {
     // update the pointer to point to the beginning of the next row
     float* rowData = (float*)(((char*)d_array) + (row * pitch));
        
    for (int column = 0; column < columnCount; column++) 
    {
      rowData[column] = 123.0; // make every value in the array 123.0
      destinationArray[(row*columnCount) + column] = rowData[column];
    }
  }
}


int main(int argc, char** argv) 
{	
  int columnCount = 15; 
  int rowCount = 10;
  float* d_array; // the device array which memory will be allocated to
  float* d_destinationArray; // the device array
  
  // allocate memory on the host
  float* h_array = new float[columnCount*rowCount];

  // the pitch value assigned by cudaMallocPitch
  // (which ensures correct data structure alignment)
  size_t pitch; 
  
  //allocated the device memory for source array
  cudaMallocPitch(&d_array, &pitch, columnCount * sizeof(float), rowCount);
  
  //allocate the device memory for destination array
  cudaMalloc(&d_destinationArray,columnCount*rowCount*sizeof(float));
  
  //call the kernel which copies values from d_array to d_destinationArray
  CopyData<<<100, 512>>>(d_array, d_destinationArray, pitch, columnCount, rowCount);

  //copy the data back to the host memory
  cudaMemcpy(h_array,
                    d_destinationArray,
                    columnCount*rowCount*sizeof(float),
                    cudaMemcpyDeviceToHost);

  //print out the values (all the values are 123.0)
  for(int i = 0 ; i < rowCount ; i++)
  {
    for(int j = 0 ; j < columnCount ; j++)
    {
      cout << "h_array[" << (i*columnCount) + j << "]=" << h_array[(i*columnCount) + j] << endl;
    }
  }
}