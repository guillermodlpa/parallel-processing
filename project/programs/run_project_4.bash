#!/bin/bash

# Modifying working directory
# I did this to adapt the script to my custom working directory structure

	cd parallel-processing/project/programs
	echo "Working directory is:"
	pwd

mpirun -npernode 8 ./project_4
#mpirun -npernode 8 ./project_4 sample/2_im1 sample/2_im2