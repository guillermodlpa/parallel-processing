#!/bin/bash

# Modifying working directory
# I did this to adapt the script to my custom working directory structure
if [ -d "parallel-processing/hw4" ]; then
	cd parallel-processing/hw4
	echo "Working directory is:"
	pwd
	echo ""
fi

mpirun -npernode 8 ./get_data
