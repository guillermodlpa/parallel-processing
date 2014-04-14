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


float test[2];

int main(int argc, char **argv) {

	/* Prototype function*/
	void gauss();

	MPI_Init(&argc, &argv);

	/* Get my process rank */
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	/* Find out how many processes are being used */
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    printf("\nProcess number %d of %d says hi\n",
            my_rank+1, p);

    test[0] = 0;
    test[1] = 0;
    
    gauss();

	MPI_Finalize();
}

/* Main function that performs the algorithms */
void gauss() {

	MPI_Status status;

    if ( my_rank == SOURCE ) {
		int i;
		test[0] = 1;
		test[1] = 2;
    	for ( i = 1; i < p; i++ ) {
    		MPI_Send( &test, 2, MPI_FLOAT, i,0, MPI_COMM_WORLD );
    	}
    }
    else
		MPI_Recv( &test, 2, MPI_FLOAT, SOURCE, 0, MPI_COMM_WORLD, &status);


	test[0] = test[0] + 0.01f;
	test[1] = test[1] + 0.02f;


	printf("\nProcess number %d of %d says: got %5.2f and %5.2f\n",
        my_rank+1, p, test[0], test[1]);


	if ( my_rank != SOURCE )
		MPI_Isend( &test, 2, MPI_FLOAT, SOURCE,0, MPI_COMM_WORLD );
	else {
		int i;
		float local_test [2];
		for ( i = 1; i < p; i++ ) {
    		MPI_Recv( &local_test, 2, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &status);
    		test[0] = test[0] + local_test[0];
    		test[1] = test[1] + local_test[1];
    	}
    	printf("\nProcess number %d of %d says: got %5.2f and %5.2f\n",
        my_rank+1, p, test[0], test[1]);
    }


}
