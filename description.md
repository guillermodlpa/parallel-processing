CS 546 Parallel and Distributed Processing
===================

This is the repository to upload assignments to the Jarvis cluster at the Computer Science department at IIT. It also includes the submissions and tests for each one of them.
All of them explore the <strong>Gaussian Elimination</strong> algorithm using different parallel programming paradigms.

String 2014 - Illinois Institute of Technology

Shared Memory (hw2)
-------------------

Homework 2 explores the shared memory model. Two programs have been coded, one using pthreads and another using OpenMP.
The results were satisfactory.

CUDA (hw3)
----------

The potential of GPU parallel processing with CUDA is explored in Homework 3. 
The results were satisfactory.

Message Passing Interface (hw4)
-------------------------------

MPI is used to achieve an algorithm based on multiple processors comunicating through message passing.
The results weren't good because the communication mechanisms proposed aren't efficient.
A correct solution would have been to use a cyclic distribution of rows at the beginning, and then the processor broadcasting the norm row would change in every iteration.

High Performance Fortran (hw5)
----------------------

Using High Performance Fortran (HPF), the same program is designed using the data parallelism paradigm.
The solution wasn't tested.
The solution proposed in the submitted PDF file contains errors, but the one in the revised (_REV.pdf) is correct.