CS 546 Parallel and Distributed Processing - Assignment 4 instructions
===================

To execute the assignment simply copy this folder in your home directory in Jarvis
That way, the structure is:

/home/<user>/parallel-processing/hw4

The folder hw4 includes two subfolders:

problem1
problem2

----------------------------------------------------

PROBLEM 1

To run problem1 get_data program, simply go to
/home/<user>/parallel-processing/hw4/problem1
and execute

$ mpicc -c get_data.c; mpicc -o get_data get_data.o
$ qsub -pe mpich 2 -P cs546_s14_project run_get_data.bash

The output will appear in /home/<user>/

---------------------------------------------------

PROBLEM 2

Same but with the gauss.c file