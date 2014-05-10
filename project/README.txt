CS 546 Project
==============

Project files
-------------

All the source code is located inside the folder "programs"
Go inside this folder with the command line

There, you will find different files:

- sample: folder with the samples given with the project

- project_0_sequential.c: sequential algorithm (not part of the project requirements)
- project_1.c: version of the program that uses MPI with Send & Recv
- project_2.c: version of the program that uses MPI with collective calls
- project_3.c: version of the program that uses MPI + OpenMP
- project_4.c: version of the program that uses task parallelism with four groups

- run_project_1.bash: script to execute project_1 in the Jarvis cluster
- run_project_2.bash: script to execute project_2 in the Jarvis cluster
- run_project_3.bash: script to execute project_3 in the Jarvis cluster
- run_project_4.bash: script to execute project_4 in the Jarvis cluster

- output_matrix: file containing the output of the last execution, 
  it has the same format as the sample files provided with the project


Execution commands and parameters
---------------------------------

To run the project_1 program, which is the one that uses MPI with Send and Recv functions:

    # If you're in the Jarvis home folder, to /parallel-processing/project/programs
    $ cd parallel-processing/project/programs

    # Compile
    $ mpicc -c project_1.c; mpicc -o project_1 project_1.o

    # Run
    $ qsub -pe mpich 2 -P cs546_s14_project run_project_1.bash

    # Check out the results, which will be located in the home directory (remember, you are in parallel-processing/project/programs)
    $ cat ../../../run*

    # To delete all outputs before starting a new job
    $ rm ../../../run*

For project_2 and project_4, the commands are similar, although the numbers must be changed. 

For project_3, you must add a flag to include OpenMP into the compilation:

    $ mpicc -c project_3.c -fopenmp; mpicc -o project_3 project_3.o -fopenmp

And here I leave a line that will compile all programs. Copy it and paste it to compile everything

	$ mpicc -c project_1.c; mpicc -o project_1 project_1.o; mpicc -c project_2.c; mpicc -o project_2 project_2.o; mpicc -c project_3.c -fopenmp; mpicc -o project_3 project_3.o -fopenmp; mpicc -c project_4.c; mpicc -o project_4 project_4.o


To modify the number of nodes, go to each of the run_project_X.bash and modify the parameter that goes with -npernode.
The programs are designed to work with 1, 2, 4 and 8 processes.
This is the flag that was modified to obtain the different execution times required by the project.

For project_3, the number of threads is directly specified in the code with the variable NUM_THREADS (line 42, current value is 2).


Inputs and outputs
------------------

It is default that all the programs will use these images:

	Im1: sample/1_im1
	Im2: sample/1_im2

You can modify those values directly in the code, or also pass them as arguments into the .bash file.
For example, to make the project_4 work with 2_im1 and 2_im2, go to run_project_4.bash and uncomment this line:

	#mpirun -npernode 8 ./project_4 sample/2_im1 sample/2_im2

The output is saved into "parallel-processing/project/output_matrix"
It has the same format than the samples provided, to allow easy comparison with the sample outputs
