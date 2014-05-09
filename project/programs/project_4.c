/*

This program is the parallel algorithm using MPI Send and Recv
We suppost that N is always going to be divisible between the number of processes

The default input is sample/1_im1 and sample/1_im2
To indicate other inputs:

   $ ./a.out [image1] [image2]

The output file is saved as "output_matrix" in the working directory

This algorithm is better explained in the document that goes with this project

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
const int N = 16;


int p, my_rank;
#define SOURCE 0

int main (int argc, char **argv) {

   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
   MPI_Comm_size(MPI_COMM_WORLD, &p);

   /* Input files */
   const char* filename1 = argc == 3 ? argv[1] : "sample/2_im1";
   const char* filename2 = argc == 3 ? argv[2] : "sample/2_im2";

   if ( my_rank==0) printf("\nCS 546 Project: MPI with Send + Recv\n");
   if ( my_rank==0) printf("CS 546 Project: Number of processors = %d\n",p);
   if ( my_rank==0) printf("CS 546 Project: using images %s, %s\n\n",filename1, filename2);

   /* Prototype functions */
   int read_matrix ( const char* filename, complex matrix[N][N] );
   int write_matrix ( const char* filename, complex matrix[N][N] );
   void c_fft1d(complex *r, int n, int isign);
   void print_matrix ( complex matrix[N][N], const char* matrixname, int rank );


   /* Variable init */
   int chunk = N / p; /* number of rows for each process */
   complex A[N][N], B[N][N], C[N][N];
   int i, j;
   complex tmp;
   double time_init, time_end, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10;
   MPI_Status status;


   /* Read files */
   read_matrix (filename1, A);
   read_matrix (filename2, B);

   print_matrix(A, "Matrix A initial", SOURCE);
   print_matrix(B, "Matrix B initial", SOURCE);

   /* Initial time */
   if ( my_rank == SOURCE )
      time_init = MPI_Wtime();


   /* Temporary to put zeros everywhere, uUseful when debugging and reading intermediate matrixes */
   if ( my_rank != SOURCE )
   for (i=0;i<N;i++)
      for (j=0;j<N;j++) { A[i][j].r = 0; A[i][j].i = 0; B[i][j].r = 0; B[i][j].i = 0;}
   

/*-------------------------------------------------------------------------------------------------------*/
   /* Divide the processors in 4 groups */ 

   int group_size = p / 4;
   int my_grp_rank;
   int P1_array[group_size], P2_array[group_size], P3_array[group_size], P4_array[group_size];

   for(i=0; i<p; i++) {
      int processor_group = i / group_size;
      switch(processor_group){
      case 0:
         P1_array[ i%group_size ] = i;
         break;
      case 1:
         P2_array[ i%group_size ] = i;
         break;
      case 2:
         P3_array[ i%group_size ] = i;
         break;
      case 3:
         P4_array[ i%group_size ] = i;
         break;
      }
   }
   
   MPI_Group world_group, P1, P2, P3, P4; 
   MPI_Comm P1_comm, P2_comm, P3_comm, P4_comm;
   //MPI_Comm P1_P2_inter;

   /* Extract the original group handle */ 
   MPI_Comm_group(MPI_COMM_WORLD, &world_group); 

   /* Create the for groups */
   int my_group = my_rank / group_size;


   if ( my_group == 0 )      { 
      
      MPI_Group_incl(world_group, p/4, P1_array, &P1);
      MPI_Comm_create( MPI_COMM_WORLD, P1, &P1_comm);
      MPI_Group_rank(P1, &my_grp_rank);
      //MPI_Intercomm_create(P1_comm, 0, MPI_COMM_WORLD, P2_array[0], 111, &P1_P2_inter);
   } 
   else if ( my_group == 1 ) { 

      MPI_Group_incl(world_group, p/4, P2_array, &P2); 
      MPI_Comm_create( MPI_COMM_WORLD, P2, &P2_comm);
      MPI_Group_rank(P2, &my_grp_rank);
      //MPI_Intercomm_create(P2_comm, 0, MPI_COMM_WORLD, P1_array[0], 111, &P1_P2_inter);
   } 
   else if ( my_group == 2 ) { 
      MPI_Group_incl(world_group, p/4, P3_array, &P3); 
      MPI_Comm_create( MPI_COMM_WORLD, P3, &P3_comm);
      MPI_Group_rank(P3, &my_grp_rank);
   } 
   else if ( my_group == 3 ) { 
      MPI_Group_incl(world_group, p/4, P4_array, &P4); 
      MPI_Comm_create( MPI_COMM_WORLD, P4, &P4_comm);
      MPI_Group_rank(P4, &my_grp_rank);
   } 


   
/*-------------------------------------------------------------------------------------------------------*/
   /* Scatter A and B to the other processes. We supose N is divisible by p */

   chunk = N / group_size;

   if ( my_rank == SOURCE ){

      // Send A to the P1 processors
      for ( i=0; i<group_size; i++ ) {
         if ( P1_array[i]==SOURCE ) continue;
         MPI_Send( &A[chunk*i][0], chunk*N, MPI_COMPLEX, P1_array[i], 0, MPI_COMM_WORLD );
      }
      // Send B to the P2 processors
      for ( i=0; i<group_size; i++ ) {
         if ( P2_array[i]==SOURCE ) continue;
         MPI_Send( &B[chunk*i][0], chunk*N, MPI_COMPLEX, P2_array[i], 0, MPI_COMM_WORLD );
      }
   }
   else {

      // Receive A because this is group P1
      if ( my_group == 0 )
         MPI_Recv( &A[chunk*my_grp_rank][0], chunk*N, MPI_COMPLEX, SOURCE, 0, MPI_COMM_WORLD, &status );
      // Receive B because this is group P2
      if ( my_group == 1 )
         MPI_Recv( &B[chunk*my_grp_rank][0], chunk*N, MPI_COMPLEX, SOURCE, 0, MPI_COMM_WORLD, &status );
   }
   if ( my_rank == SOURCE ) t1 = MPI_Wtime();

   
/*-------------------------------------------------------------------------------------------------------*/
   /* Apply 1D FFT in all rows of A, in group P1 */

   if ( my_group == 0 )
      for ( i=chunk*my_grp_rank; i<chunk*(my_grp_rank+1); i++ )
         c_fft1d(A[i], N, -1);

/*-------------------------------------------------------------------------------------------------------*/
   /* Apply 1D FFT in all rows of B, in group P2 */
   
   if ( my_group == 1 )
      for ( i=chunk*my_grp_rank; i<chunk*(my_grp_rank+1); i++ )
         c_fft1d(B[i], N, -1);


   if ( my_rank == SOURCE ) t2 = MPI_Wtime();

/*-------------------------------------------------------------------------------------------------------*/
   /* Gather the row FFTs from A into the first processor of P1 for sequential trasposition */

   if ( my_group == 0 ) {
      if ( my_grp_rank == 0 ) {
         for ( i=1; i<group_size; i++ ) {
            MPI_Recv( &A[chunk*i][0], chunk*N, MPI_COMPLEX, i, 0, P1_comm, &status );
         }
         
      }
      else 
         MPI_Send( &A[chunk*my_grp_rank][0], chunk*N, MPI_COMPLEX, 0, 0, P1_comm );
   }

/*-------------------------------------------------------------------------------------------------------*/
   /* Gather the row FFTs from B into the first processor of P2 for sequential trasposition */

   if ( my_group == 1 ) {
      if ( my_grp_rank == 0 ) {
         for ( i=1; i<group_size; i++ ) {
            MPI_Recv( &B[chunk*i][0], chunk*N, MPI_COMPLEX, i, 0, P2_comm, &status );
         }
         
      }
      else 
         MPI_Send( &B[chunk*my_grp_rank][0], chunk*N, MPI_COMPLEX, 0, 0, P2_comm );
   }

/*-------------------------------------------------------------------------------------------------------*/
   /* Traspose matrix A in P1's main process */

   if ( my_group == 0 && my_grp_rank == 0 ) {
      for (i=0;i<N;i++) {
         for (j=i;j<N;j++) {
            tmp = A[i][j];
            A[i][j] = A[j][i];
            A[j][i] = tmp;
         }
      }
   }

   print_matrix(A, "Matrix A before receiving tranposition",1);

/*-------------------------------------------------------------------------------------------------------*/
   /* Traspose matrix B in P2's main process */
   
   if ( my_group == 1 && my_grp_rank == 0 ) {
      for (i=0;i<N;i++) {
         for (j=i;j<N;j++) {
            tmp = B[i][j];
            B[i][j] = B[j][i];
            B[j][i] = tmp;
         }
      }
   }


/*-------------------------------------------------------------------------------------------------------*/
   /* Scatter the transposed A in the group P1 */

   if ( my_group == 0 ) {
      if ( my_grp_rank == 0 ) {
         for ( i=1; i<group_size; i++ ) {
            MPI_Send( &A[chunk*i][0], chunk*N, MPI_COMPLEX, i, 0, P1_comm );
         }
         
      }
      else 
         MPI_Recv( &A[chunk*my_grp_rank][0], chunk*N, MPI_COMPLEX, 0, 0, P1_comm, &status );
   }

   print_matrix(A, "Matrix A after receiving transposition",1);

/*-------------------------------------------------------------------------------------------------------*/
   /* Scatter the transposed B in the group P2 */

   if ( my_group == 1 ) {
      if ( my_grp_rank == 0 ) {
         for ( i=1; i<group_size; i++ ) {
            MPI_Send( &B[chunk*i][0], chunk*N, MPI_COMPLEX, i, 0, P2_comm );
         }
         
      }
      else 
         MPI_Recv( &B[chunk*my_grp_rank][0], chunk*N, MPI_COMPLEX, 0, 0, P2_comm, &status );
   }

/*-------------------------------------------------------------------------------------------------------*/
   /* Apply 1D FFT in all rows of A, in group P1. This are actually columns of the original A */

   if ( my_group == 0 )
      for ( i=chunk*my_grp_rank; i<chunk*(my_grp_rank+1); i++ )
         c_fft1d(A[i], N, -1);

/*-------------------------------------------------------------------------------------------------------*/
   /* Apply 1D FFT in all rows of B, in group P2. This are actually columns of the original B */

   if ( my_group == 1 )
      for ( i=chunk*my_grp_rank; i<chunk*(my_grp_rank+1); i++ )
         c_fft1d(B[i], N, -1);


   //print_matrix(A, "Matrix A after second fft",0);
   print_matrix(A, "Matrix A after second fft",1);
   //print_matrix(B, "Matrix B after second fft",2);
   //print_matrix(B, "Matrix B after second fft",3);


/*-------------------------------------------------------------------------------------------------------*/
   /* Gather A and B into the P3 processor */
   /* All the processors in P1 and P2 will send it to the first processor in P3 using the global communicator */

/*
   if ( my_group == 0 )
      MPI_Send ( &A[chunk*my_grp_rank][0], chunk*N, MPI_COMPLEX, P3_array[0], 0, MPI_COMM_WORLD );
   else if ( my_group == 1 )
      MPI_Send ( &B[chunk*my_grp_rank][0], chunk*N, MPI_COMPLEX, P3_array[0], 0, MPI_COMM_WORLD );

   else if ( my_group == 2 && my_grp_rank == 0 ) {

      for ( i=0; i<group_size; i++ )
         MPI_Recv( &A[chunk*i][0], chunk*N, MPI_COMPLEX, P1_array[i], 0, MPI_COMM_WORLD, &status );
      for ( i=0; i<group_size; i++ )
         MPI_Recv( &B[chunk*i][0], chunk*N, MPI_COMPLEX, P2_array[i], 0, MPI_COMM_WORLD, &status );
   }
   */

   chunk = N / p;
/*-------------------------------------------------------------------------------------------------------*/
   /* Gather A and B to the source processor */
   if ( my_rank == SOURCE ){
      for ( i=0; i<p; i++ ) {
         if ( i==SOURCE ) continue; /* Source process doesn't send to itself */

         MPI_Recv( &A[chunk*i][0], chunk*N, MPI_COMPLEX, i, 0, MPI_COMM_WORLD, &status );
         MPI_Recv( &B[chunk*i][0], chunk*N, MPI_COMPLEX, i, 0, MPI_COMM_WORLD, &status );
      }
   }
   else {
      MPI_Send( &A[chunk*my_rank][0], chunk*N, MPI_COMPLEX, SOURCE, 0, MPI_COMM_WORLD );
      MPI_Send( &B[chunk*my_rank][0], chunk*N, MPI_COMPLEX, SOURCE, 0, MPI_COMM_WORLD );
   }
   if ( my_rank == SOURCE ) t3 = MPI_Wtime();

   //print_matrix(A, "Matrix A after recv");
   //print_matrix(B, "Matrix B after recv");

/*-------------------------------------------------------------------------------------------------------*/
   /* Transpose matrixes sequentially */
   if ( my_rank == SOURCE ) {
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
      t4 = MPI_Wtime();
   }
   //print_matrix(A, "Matrix A after traspose");
   //print_matrix(B, "Matrix B after traspose");



/*-------------------------------------------------------------------------------------------------------*/
   /* Scatter A and B to the other processes. We supose N is divisible by p */
   if ( my_rank == SOURCE ){
      for ( i=0; i<p; i++ ) {
         if ( i==SOURCE ) continue; /* Source process doesn't send to itself */

         MPI_Send( &A[chunk*i][0], chunk*N, MPI_COMPLEX, i, 0, MPI_COMM_WORLD );
         MPI_Send( &B[chunk*i][0], chunk*N, MPI_COMPLEX, i, 0, MPI_COMM_WORLD );
      }
   }
   else {
      MPI_Recv( &A[chunk*my_rank][0], chunk*N, MPI_COMPLEX, SOURCE, 0, MPI_COMM_WORLD, &status );
      MPI_Recv( &B[chunk*my_rank][0], chunk*N, MPI_COMPLEX, SOURCE, 0, MPI_COMM_WORLD, &status );
   }
   if ( my_rank == SOURCE ) t5 = MPI_Wtime();

   //print_matrix(A, "Matrix A after nothing");
   //print_matrix(B, "Matrix B after nothing");

/*-------------------------------------------------------------------------------------------------------*/

   /* Apply 1D FFT in all rows of A and B */
   for (i= chunk*my_rank ;i< chunk*(my_rank+1);i++) {
         c_fft1d(A[i], N, -1);
         c_fft1d(B[i], N, -1);
   }

   //print_matrix(A, "Matrix A after col fft");
   //print_matrix(B, "Matrix B after col fft");

   /* Transpose matrixes */
   /* Not necessary if we remove a later traspose */

/*-------------------------------------------------------------------------------------------------------*/

   //print_matrix(C, "Matrix C pre mult");

   /* Point to point multiplication */
   for (i= chunk*my_rank ;i< chunk*(my_rank+1);i++) {
      for (j=0;j<N;j++) {
         C[i][j].r = A[i][j].r*B[i][j].r - A[i][j].i*B[i][j].i;
         C[i][j].i = A[i][j].r*B[i][j].i + A[i][j].i*B[i][j].r;
      }
   }

   //print_matrix(C, "Matrix C after mult");

/*-------------------------------------------------------------------------------------------------------*/
   /* Inverse 1D FFT in all rows of C */
   for (i= chunk*my_rank ;i< chunk*(my_rank+1);i++) {
      c_fft1d(C[i], N, 1);
   }
   if ( my_rank == SOURCE ) t6 = MPI_Wtime();

   //print_matrix(C, "Matrix C after fft");

/*-------------------------------------------------------------------------------------------------------*/
   /* Gather the fragments of C to the source processor */
   if ( my_rank == SOURCE ){
      for ( i=0; i<p; i++ ) {
         if ( i==SOURCE ) continue; /* Source process doesn't receive from itself */

         MPI_Recv( &C[chunk*i][0], chunk*N, MPI_COMPLEX, i, 0, MPI_COMM_WORLD, &status );
      }
   }
   else
      MPI_Send( &C[chunk*my_rank][0], chunk*N, MPI_COMPLEX, SOURCE, 0, MPI_COMM_WORLD );
   if ( my_rank == SOURCE ) t7 = MPI_Wtime();

   //print_matrix(C, "Matrix C after recv");

/*-------------------------------------------------------------------------------------------------------*/
   /* Transpose C sequentially */
   if ( my_rank == SOURCE ) {
      for (i=0;i<N;i++) {
         for (j=i;j<N;j++) {
            tmp = C[i][j];
            C[i][j] = C[j][i];
            C[j][i] = tmp;
         }
      }
      t8 = MPI_Wtime();
   }

/*-------------------------------------------------------------------------------------------------------*/
   /* Scatter C to the other processes */
   if ( my_rank == SOURCE ){
      for ( i=0; i<p; i++ ) {
         if ( i==SOURCE ) continue; /* Source process doesn't receive from itself */

         MPI_Send( &C[chunk*i][0], chunk*N, MPI_COMPLEX, i, 0, MPI_COMM_WORLD );
      }
   }
   else
      MPI_Recv( &C[chunk*my_rank][0], chunk*N, MPI_COMPLEX, SOURCE, 0, MPI_COMM_WORLD, &status );
   if ( my_rank == SOURCE ) t9 = MPI_Wtime();

/*-------------------------------------------------------------------------------------------------------*/
   /* Inverse 1D FFT in all columns of C */
   for (i= chunk*my_rank ;i< chunk*(my_rank+1);i++) {
      c_fft1d(C[i], N, 1);
   }
   if ( my_rank == SOURCE ) t10 = MPI_Wtime();

   /* Transpose C */
   /* It is not necessary if we remove the other traspose */


/*-------------------------------------------------------------------------------------------------------*/
   /* Gather the fragments of C to the source processor */
   if ( my_rank == SOURCE ){
      for ( i=0; i<p; i++ ) {
         if ( i==SOURCE ) continue; /* Source process doesn't receive from itself */

         MPI_Recv( &C[chunk*i][0], chunk*N, MPI_COMPLEX, i, 0, MPI_COMM_WORLD, &status );
      }
   }
   else
      MPI_Send( &C[chunk*my_rank][0], chunk*N, MPI_COMPLEX, SOURCE, 0, MPI_COMM_WORLD );

/*-------------------------------------------------------------------------------------------------------*/
   /* Final time */
   if ( my_rank == SOURCE )
      time_end = MPI_Wtime();

   print_matrix(C, "Matrix C final", SOURCE);
   if ( my_rank==0) printf("C[0][0].r     = %e\n", C[0][0].r);
   if ( my_rank==0) printf("C[N-1][N-1].r = %e\n", C[N-1][N-1].r);

   /* Write output file */
   write_matrix("output_matrix", C);

   if ( my_rank==0) printf("\nCS 546 Project: done\n");
   if ( my_rank==0) {

      double tcomputation = (t2-t1) + (t4-t3) + (t6-t5) + (t8-t7) + (t10-t9);
      double tcommunication = (t1-time_init) + (t3-t2) + (t5-t4) + (t7-t6) + (t9-t8) + (time_end-t10);

      printf("CS 546 Project: total time spent is %f ms\n", (time_end-time_init) * 1000 );
      printf("CS 546 Project: time computation is  %f ms\n", tcomputation * 1000 );
      printf("CS 546 Project: time communication is  %f ms\n", tcommunication * 1000 );
   }

   MPI_Finalize();
}




/*-------------------------------------------------------------------------------------------------------*/

/* Reads the matrix from tha file and inserts the values in the real part */
/* The complex part is left to 0 */
int read_matrix ( const char* filename, complex matrix[N][N] ) {
   if ( my_rank == SOURCE ) {
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
}

/* Write the real part of the result matrix */
int write_matrix ( const char* filename, complex matrix[N][N] ) {
   if ( my_rank == SOURCE ) {
      int i, j;
      FILE *fp = fopen(filename,"w");

      for (i=0;i<N;i++) {
         for (j=0;j<N;j++)
            fprintf(fp,"   %e",matrix[i][j].r);
         fprintf(fp,"\n");
      };

      fclose(fp);
   }
}

/* Print the matrix if its size is no more than 32x32 */
/* Rank is the processor that should print this */
void print_matrix ( complex matrix[N][N], const char* matrixname, int rank ) {
   if ( my_rank == rank ) {
      if ( N<33 ) {
         int i, j;
         printf("%s by process #%d\n",matrixname, rank);
         for (i=0;i<N;i++){
            for (j=0;j<N;j++) {
              printf("(%.1f,%.1f) ", matrix[i][j].r,matrix[i][j].i);
           }printf("\n");
         }printf("\n");
      }
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
