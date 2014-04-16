#!/bin/bash

# Execute instructions
#
#   $ qsub -pe mpich 2 -P cs546_s14_project run_get_data.bash
#
#

# Modifying working directory
# I did this to adapt the script to my custom working directory structure

	cd parallel-processing/hw4/1
	echo "Working directory is:"
	pwd

mpirun -npernode 8 ./get_data
