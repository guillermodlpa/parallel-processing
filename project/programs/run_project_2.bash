#!/bin/bash

# Modifying working directory
# I did this to adapt the script to my custom working directory structure

	cd parallel-processing/project/programs
	echo "Working directory is:"
	pwd

mpirun -npernode 2 ./project_2
