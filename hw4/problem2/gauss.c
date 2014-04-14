/* ------------------ HW3 --------------------- */

/* Guillermo de la Puente - A20314328                */
/* CS 546 - Parallel and Distributed Processing      */
/* Homework 4                                        */
/* Part 2/2                                          */
/* Gaussian elimination without pivoting with MPI    */
/*                                                   */
/* 2014 Spring semester                              */
/* Professor Zhiling Lan                             */
/* TA Eduardo Berrocal                               */
/* Illinois Institute of Technology                  */


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
#include <time.h>

#include <mpi.h>

#define MAXN 4000 /* Max value of N */
int N; /* Matrix size */
#define DIVFACTOR 327680000.0f

#define SOURCE 0

/* My process rank           */
int my_rank;
/* The number of processes   */
int p; 


int main(int argc, char **argv) {

	MPI_Init(&argc, &argv);

	/* Get my process rank */
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	/* Find out how many processes are being used */
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    printf("\nProcess number %d of %d says hi\n",
            my_rank+1, p);

    int test = 0;
    if ( my_rank == SOURCE ) {
		int i;
		test = 1;
    	for ( i = 1; i < p; i++ ) {
    		MPI_Send( &test, 1, MPI_INT, i,0, MPI_COMM_WORLD );
    	}
    }
    else
		MPI_Recv( &test, 1, MPI_INT, SOURCE, 0, MPI_COMM_WORLD, &status);

	printf("\nProcess number %d of %d says: got %d\n",
        my_rank+1, p, i);


	MPI_Finalize();
}


