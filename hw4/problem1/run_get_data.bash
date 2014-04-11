#!/bin/bash

# Modifying working directory
# I did this to adapt the script to my custom working directory structure

	cd parallel-processing/hw4/problem2
	echo "Working directory is:"
	pwd

mpirun -npernode 8 ./get_data
