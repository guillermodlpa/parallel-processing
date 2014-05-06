/*

This program is the sequential algorithm

*/
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <mpi.h>

/* Given with the fft function in the assignment */
typedef struct {float r; float i;} complex;
static complex ctmp;
#define C_SWAP(a,b) {ctmp=(a);(a)=(b);(b)=ctmp;}


/* Size of matrix (NxN) */
const int N = 512;


int p, my_rank;

int main (int argc, char **argv) {

   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
   MPI_Comm_size(MPI_COMM_WORLD, &p);

   if ( my_rank==0) printf("CS 546 Project: MPI with Send + Recv\n");

   /* Input files */
   const char* filename1 = argc == 3 ? argv[1] : "sample/1_im1";
   const char* filename2 = argc == 3 ? argv[2] : "sample/1_im2";

   /* Prototype functions */
   int read_matrix ( const char* filename, complex matrix[N][N] );
   void c_fft1d(complex *r, int n, int isign);
   void print_matrix ( complex matrix[N][N], const char* matrixname );


   /* Variable init */
   complex A[N][N], B[N][N], C[N][N];
   int i, j;
   complex tmp;

   /* Read files */
   read_matrix (filename1, A);
   read_matrix (filename2, B);

   print_matrix(A, "Matrix A");
   print_matrix(B, "Matrix B");


   /* Apply 1D FFT in all rows of A and B */
   for (i=0;i<N;i++) {
      c_fft1d(A[i], N, -1);
      c_fft1d(B[i], N, -1);
   }

   /* Transpose matrixes */
   for (i=0;i<N;i++) {
      for (j=i;j<N;j++) {
         tmp = A[i][j];
         A[i][j] = A[j][i];
         A[j][i] = tmp;

         tmp = B[i][j];
         B[i][j] = B[j][i];
         B[j][i] = tmp;
      }
   }

   /* Apply 1D FFT in all rows of transposed A and transposed B */
   for (i=0;i<N;i++) {
      c_fft1d(A[i], N, -1);
      c_fft1d(B[i], N, -1);
   }

   /* Transpose matrixes */
   /* Not necessary if we remove a later traspose */


   /* Point to point multiplication */
   for (i=0;i<N;i++) {
      for (j=0;j<N;j++) {
         C[i][j].r = A[i][j].r*B[i][j].r - A[i][j].i*B[i][j].i;
         C[i][j].i = A[i][j].r*B[i][j].i + A[i][j].i*B[i][j].r;
      }
   }

   /* Inverse 1D FFT in all rows of C */
   for (i=0;i<N;i++) {
      c_fft1d(C[i], N, 1);
   }

   /* Transpose C */
   for (i=0;i<N;i++) {
      for (j=i;j<N;j++) {
         tmp = C[i][j];
         C[i][j] = C[j][i];
         C[j][i] = tmp;
      }
   }

   /* Inverse 1D FFT in all columns of C */
   for (i=0;i<N;i++) {
      c_fft1d(C[i], N, 1);
   }

   /* Transpose C */
   /* It is not necessary if we remove the other traspose */

   print_matrix(C, "Matrix C");

   if ( my_rank==0) printf("CS 546 Project: done\n");
}




/* Reads the matrix from tha file and inserts the values in the real part */
/* The complex part is left to 0 */
int read_matrix ( const char* filename, complex matrix[N][N] ) {
   int i, j;
   FILE *fp = fopen(filename,"r");

   if ( !fp ) {
      printf("Error. This file couldn't be read because it doesn't exist: %s\n", filename);
      exit(1);
   }

   for (i=0;i<N;i++)
      for (j=0;j<N;j++) {
         fscanf(fp,"%g",&matrix[i][j].r);
         matrix[i][j].i = 0;
      }
   fclose(fp);
}


/* Print the matrix if its size is no more than 32x32 */
void print_matrix ( complex matrix[N][N], const char* matrixname ) {
   if ( N<33 ) {
      int i, j;
      printf("%s\n",matrixname);
      for (i=0;i<N;i++){
         for (j=0;j<N;j++) {
           printf("%f ", matrix[i][j].r);
        }printf("\n");
      }printf("\n");
   }
}



/*
 ------------------------------------------------------------------------
 FFT1D            c_fft1d(r,i,-1)
 Inverse FFT1D    c_fft1d(r,i,+1)
 ------------------------------------------------------------------------
*/
/* ---------- FFT 1D
   This computes an in-place complex-to-complex FFT
   r is the real and imaginary arrays of n=2^m points.
   isign = -1 gives forward transform
   isign =  1 gives inverse transform
*/

void c_fft1d(complex *r, int      n, int      isign)
{
   int     m,i,i1,j,k,i2,l,l1,l2;
   float   c1,c2,z;
   complex t, u;

   if (isign == 0) return;

   /* Do the bit reversal */
   i2 = n >> 1;
   j = 0;
   for (i=0;i<n-1;i++) {
      if (i < j)
         C_SWAP(r[i], r[j]);
      k = i2;
      while (k <= j) {
         j -= k;
         k >>= 1;
      }
      j += k;
   }

   /* m = (int) log2((double)n); */
   for (i=n,m=0; i>1; m++,i/=2);

   /* Compute the FFT */
   c1 = -1.0;
   c2 =  0.0;
   l2 =  1;
   for (l=0;l<m;l++) {
      l1   = l2;
      l2 <<= 1;
      u.r = 1.0;
      u.i = 0.0;
      for (j=0;j<l1;j++) {
         for (i=j;i<n;i+=l2) {
            i1 = i + l1;

            /* t = u * r[i1] */
            t.r = u.r * r[i1].r - u.i * r[i1].i;
            t.i = u.r * r[i1].i + u.i * r[i1].r;

            /* r[i1] = r[i] - t */
            r[i1].r = r[i].r - t.r;
            r[i1].i = r[i].i - t.i;

            /* r[i] = r[i] + t */
            r[i].r += t.r;
            r[i].i += t.i;
         }
         z =  u.r * c1 - u.i * c2;

         u.i = u.r * c2 + u.i * c1;
         u.r = z;
      }
      c2 = sqrt((1.0 - c1) / 2.0);
      if (isign == -1) /* FWD FFT */
         c2 = -c2;
      c1 = sqrt((1.0 + c1) / 2.0);
   }

   /* Scaling for inverse transform */
   if (isign == 1) {       /* IFFT*/
      for (i=0;i<n;i++) {
         r[i].r /= n;
         r[i].i /= n;
      }
   }
}
